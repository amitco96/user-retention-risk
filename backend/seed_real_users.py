"""
Seed PostgreSQL with real users from the medium Sparkify dataset.

This script:
  1. Wipes the existing synthetic users / events / risk_scores.
  2. Stream-parses ml/data/medium-sparkify-event-data.json (242 MB JSONL),
     groups events per Sparkify userId, and inserts them as real Users +
     Events using the project's SQLAlchemy 2.0 async ORM.
  3. Calls POST /pipeline/rescore-all so the dashboard reflects a realistic
     risk-tier distribution.

Reality notes
-------------
1. The medium dataset contains roughly ~448 distinct real users after dropping
   anonymous rows (empty userId). The original brief asked for ~2500 users with
   ~625 per tier; that simply isn't achievable from this dataset and
   synthesizing duplicates / forcing tier balance via subsampling would defeat
   the whole purpose of switching from synthetic to real data. We load every
   real user and report the natural distribution honestly.
2. Sparkify's events are vintage 2018; against a present-day clock every user
   would look ~2700 days inactive, which saturates the model's recency feature
   and pushes everyone to critical. We re-anchor timestamps so the latest
   event lands at "now" while preserving relative spacing — see ``delta_ms``
   below.

Run inside the container:

    docker exec retention-app sh -c \\
      "cd /app && PYTHONPATH=/app python backend/seed_real_users.py"
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy import delete

from backend.app.db.models import Base, Event, RiskScore, User
from backend.app.db.session import async_session_maker, engine

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(
    os.environ.get(
        "SPARKIFY_MEDIUM_PATH",
        "/app/ml/data/medium-sparkify-event-data.json",
    )
)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_PIPELINE_SECRET = os.environ.get("API_PIPELINE_SECRET")

EVENT_COMMIT_BATCH = 5000


# ----------------------------------------------------------------------------
# Sparkify page -> (event_type, metadata) mapping
# ----------------------------------------------------------------------------
def map_page_to_event(page: str) -> tuple[str, dict[str, Any]]:
    """Map a Sparkify `page` value to our (event_type, event_metadata) pair.

    The trained model relies on counts of NextSong / Thumbs Up / Thumbs Down /
    Add to Playlist; everything else (Home, Logout, Roll Advert, Settings,
    Cancellation Confirmation, ...) is recorded as a generic 'login' so it
    still counts toward session reconstruction without affecting feature
    counters.
    """
    if page == "NextSong":
        return "feature_used", {"page": "NextSong"}
    if page == "Thumbs Up":
        return "support_ticket", {"sentiment": "positive", "page": "Thumbs Up"}
    if page == "Thumbs Down":
        return "support_ticket", {"sentiment": "negative", "page": "Thumbs Down"}
    if page == "Add to Playlist":
        return "feature_used", {
            "feature_name": "playlist",
            "page": "Add to Playlist",
        }
    return "login", {"page": page}


# ----------------------------------------------------------------------------
# Streaming parse of the Sparkify JSONL file
# ----------------------------------------------------------------------------
def stream_sparkify(path: Path):
    """Yield one parsed JSON object per non-empty line."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_user_records(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Read the file once and group events by userId.

    Returns ``(users_dict, max_ts_ms)`` where ``users_dict`` is keyed by
    Sparkify userId (str) with:
        {
          "level":        latest seen subscription level,
          "registration": registration ms epoch (or None),
          "churned":      bool — saw 'Cancellation Confirmation',
          "events":       list[(ts_ms, page)],
        }
    and ``max_ts_ms`` is the largest event timestamp seen anywhere in the file.
    """
    users: dict[str, dict[str, Any]] = {}
    total_rows = 0
    dropped_anon = 0
    max_ts_ms = 0

    for row in stream_sparkify(path):
        total_rows += 1
        uid = row.get("userId")
        if uid is None or uid == "":
            dropped_anon += 1
            continue

        page = row.get("page", "")
        ts = row.get("ts")
        if ts is None:
            continue

        rec = users.get(uid)
        if rec is None:
            rec = {
                "level": row.get("level", "free"),
                "registration": row.get("registration"),
                "churned": False,
                "events": [],
            }
            users[uid] = rec

        # Track latest seen level (paid > free is sticky enough; just keep
        # whatever the user currently is when we see them).
        if row.get("level"):
            rec["level"] = row["level"]
        if rec["registration"] is None and row.get("registration") is not None:
            rec["registration"] = row["registration"]
        if page == "Cancellation Confirmation":
            rec["churned"] = True

        rec["events"].append((ts, page))
        if ts > max_ts_ms:
            max_ts_ms = ts

    logger.info(
        "Parsed %d rows; %d distinct users; %d anonymous rows dropped",
        total_rows,
        len(users),
        dropped_anon,
    )
    return users, max_ts_ms


# ----------------------------------------------------------------------------
# DB ops
# ----------------------------------------------------------------------------
async def ensure_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def wipe_existing(session) -> None:
    """Delete in FK-safe order: risk_scores -> events -> users."""
    logger.info("Wiping existing risk_scores / events / users ...")
    await session.execute(delete(RiskScore))
    await session.execute(delete(Event))
    await session.execute(delete(User))
    await session.commit()


def ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


async def load_users_and_events(
    session,
    sparkify_users: dict[str, dict[str, Any]],
    delta_ms: int,
) -> tuple[int, int, int]:
    """Insert users + events. Returns (users_inserted, events_inserted, churned_count).

    ``delta_ms`` is added to every Sparkify timestamp before insertion so the
    dataset's relative spacing is preserved while the latest event lands at
    "now" — without this shift the 2018-vintage Sparkify data leaves every
    user looking ~2700 days inactive, which saturates the model's recency
    feature and pushes everyone into the critical tier.
    """
    users_inserted = 0
    events_inserted = 0
    churned_count = 0
    pending_events_in_batch = 0

    fallback_signup = datetime.now(timezone.utc)

    for sparkify_uid, rec in sparkify_users.items():
        if not rec["events"]:
            continue

        plan_type = "pro" if rec["level"] == "paid" else "free"

        if rec["registration"]:
            try:
                signup_date = ms_to_utc(int(rec["registration"]) + delta_ms)
            except (TypeError, ValueError, OSError):
                signup_date = fallback_signup
        else:
            earliest_ts = min(ev[0] for ev in rec["events"])
            signup_date = ms_to_utc(int(earliest_ts) + delta_ms)

        user = User(
            email=f"sparkify_{sparkify_uid}@example.com",
            plan_type=plan_type,
            signup_date=signup_date,
        )
        session.add(user)
        await session.flush()
        users_inserted += 1
        if rec["churned"]:
            churned_count += 1

        for ts_ms, page in rec["events"]:
            event_type, metadata = map_page_to_event(page)
            try:
                occurred_at = ms_to_utc(int(ts_ms) + delta_ms)
            except (TypeError, ValueError, OSError):
                continue
            session.add(
                Event(
                    user_id=user.id,
                    event_type=event_type,
                    event_metadata=metadata,
                    occurred_at=occurred_at,
                )
            )
            events_inserted += 1
            pending_events_in_batch += 1

            if pending_events_in_batch >= EVENT_COMMIT_BATCH:
                await session.commit()
                pending_events_in_batch = 0
                logger.info(
                    "  committed batch — users=%d events=%d",
                    users_inserted,
                    events_inserted,
                )

    if pending_events_in_batch > 0:
        await session.commit()

    return users_inserted, events_inserted, churned_count


# ----------------------------------------------------------------------------
# Trigger rescore endpoint
# ----------------------------------------------------------------------------
async def trigger_rescore() -> dict[str, Any]:
    if not API_PIPELINE_SECRET:
        raise RuntimeError(
            "API_PIPELINE_SECRET not set; cannot call /pipeline/rescore-all"
        )
    url = f"{API_BASE_URL}/pipeline/rescore-all"
    headers = {"Authorization": f"Bearer {API_PIPELINE_SECRET}"}
    logger.info("Calling %s ...", url)
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


# ----------------------------------------------------------------------------
# Tier band check
# ----------------------------------------------------------------------------
def realism_check(summary: dict[str, Any]) -> str:
    total = max(1, summary.get("scored", 0))
    pct = {
        tier: round(100.0 * summary.get(tier, 0) / total, 1)
        for tier in ("critical", "high", "medium", "low")
    }
    target = {
        "critical": (5, 10),
        "high": (15, 20),
        "medium": (30, 40),
    }
    notes = []
    for tier, (lo, hi) in target.items():
        v = pct[tier]
        marker = "in band" if lo <= v <= hi else "off-target"
        notes.append(f"{tier}={v}% ({marker} {lo}-{hi}%)")
    notes.append(f"low={pct['low']}% (remainder)")
    return " | ".join(notes)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
async def main() -> None:
    t0 = time.monotonic()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sparkify file not found: {DATA_PATH}")

    logger.info("Reading Sparkify dataset: %s", DATA_PATH)
    sparkify_users, max_ts_ms = build_user_records(DATA_PATH)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    delta_ms = now_ms - max_ts_ms if max_ts_ms else 0
    logger.info(
        "Re-anchoring timestamps: max event ts=%s, shift delta=%.1f days",
        ms_to_utc(max_ts_ms).isoformat() if max_ts_ms else "n/a",
        delta_ms / 1000.0 / 86400.0,
    )

    await ensure_tables()

    async with async_session_maker() as session:
        await wipe_existing(session)
        users_inserted, events_inserted, churned = await load_users_and_events(
            session, sparkify_users, delta_ms
        )

    elapsed = time.monotonic() - t0
    churn_rate = (
        round(100.0 * churned / users_inserted, 2) if users_inserted else 0.0
    )
    logger.info(
        "LOAD COMPLETE: users=%d events=%d churned=%d (%.2f%%) elapsed=%.1fs",
        users_inserted,
        events_inserted,
        churned,
        churn_rate,
        elapsed,
    )

    logger.info("Triggering /pipeline/rescore-all ...")
    summary = await trigger_rescore()
    logger.info("Rescore response: %s", json.dumps(summary, indent=2))

    logger.info("Realism check: %s", realism_check(summary))


if __name__ == "__main__":
    asyncio.run(main())
