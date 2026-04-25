"""
Internal nightly pipeline endpoint.

POST /pipeline/rescore-all
- Loads every user, runs the ML model on their event history,
  and replaces their row in `risk_scores` with a fresh snapshot.
- Bearer-token protected via API_PIPELINE_SECRET.
- Intended to be called by the n8n nightly workflow (no rate limit).
"""

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth import verify_pipeline_token
from backend.app.db.models import Event, RiskScore, User
from backend.app.db.session import get_db
from backend.app.ml.features import extract_user_features
from backend.app.ml.model import predict

logger = logging.getLogger(__name__)

router = APIRouter()


class RescoreSummary(BaseModel):
    scored: int
    critical: int
    high: int
    medium: int
    low: int
    duration_seconds: float


def _build_reason(days_inactive: int, top_drivers: list[str]) -> str:
    driver = top_drivers[0] if top_drivers else "unknown"
    return f"User inactive for {days_inactive}d; top driver: {driver}"


def _build_action(risk_score: int) -> str:
    if risk_score >= 70:
        return "Send re-engagement email with personalized offer"
    return "Monitor activity over next 2 weeks"


@router.post(
    "/rescore-all",
    response_model=RescoreSummary,
    status_code=200,
    tags=["pipeline"],
    summary="Rescore every user; replace their risk_scores row",
    dependencies=[Depends(verify_pipeline_token)],
)
async def rescore_all(db: AsyncSession = Depends(get_db)) -> RescoreSummary:
    started_at = time.monotonic()
    counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    scored_total = 0

    try:
        users_result = await db.execute(select(User))
        users = users_result.scalars().all()

        for user in users:
            try:
                ev_result = await db.execute(
                    select(Event)
                    .where(Event.user_id == user.id)
                    .order_by(Event.occurred_at)
                )
                events = ev_result.scalars().all()
                if not events:
                    continue

                features = extract_user_features(events)
                pred = predict(str(user.id), features)

                now = datetime.now(timezone.utc)
                last_event_at = events[-1].occurred_at
                if last_event_at.tzinfo is None:
                    last_event_at = last_event_at.replace(tzinfo=timezone.utc)
                days_inactive = (now - last_event_at).days

                await db.execute(
                    delete(RiskScore).where(RiskScore.user_id == user.id)
                )
                db.add(
                    RiskScore(
                        user_id=user.id,
                        risk_score=pred.risk_score,
                        risk_tier=pred.risk_tier,
                        top_drivers=json.dumps(pred.top_drivers),
                        shap_values=pred.shap_values,
                        claude_reason=_build_reason(days_inactive, pred.top_drivers),
                        claude_action=_build_action(pred.risk_score),
                        model_version=pred.model_version,
                        scored_at=now,
                    )
                )

                counts[pred.risk_tier] = counts.get(pred.risk_tier, 0) + 1
                scored_total += 1

                if scored_total % 100 == 0:
                    await db.commit()

            except Exception as e:
                logger.warning("Skipped user %s during rescore: %s", user.id, e)

        await db.commit()

    except Exception as e:
        logger.error("rescore-all failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Rescore pipeline failed")

    duration = time.monotonic() - started_at
    return RescoreSummary(
        scored=scored_total,
        critical=counts["critical"],
        high=counts["high"],
        medium=counts["medium"],
        low=counts["low"],
        duration_seconds=round(duration, 3),
    )
