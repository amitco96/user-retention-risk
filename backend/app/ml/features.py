"""
Feature engineering utilities for live user data (PostgreSQL events).

Wraps ml/feature_engineering.py to extract features for a specific user
from the live event database. Returns all 8 features expected by the
model trained on the medium Sparkify dataset.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional


# 30-minute gap threshold for session segmentation (>30min idle = new session)
SESSION_GAP_SECONDS = 30 * 60


def extract_user_features(
    user_events: list,
    observation_date: Optional[datetime] = None,
    plan_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Extract feature dict for a single user from their events.

    Args:
        user_events: List of event-like objects with attributes:
            - event_type: str (login, feature_used, support_ticket)
            - event_metadata: dict (optional)
            - occurred_at: datetime
        observation_date: Reference date for recency-style features
            (defaults to datetime.utcnow()).
        plan_type: Optional subscription plan string from the User record
            ("free" / "starter" / "pro" / "enterprise"). Defaults to "free"
            when not provided. Encoded as 0 for "free", 1 otherwise.

    Returns:
        Dict[str, float] with all 8 keys required by the trained model:
            - days_since_last_activity
            - session_count_30d
            - songs_played_total
            - thumbs_up_count
            - thumbs_down_count
            - add_to_playlist_count
            - avg_session_duration_min
            - subscription_level

    Raises:
        ValueError: If user_events is empty.
    """
    if not user_events:
        raise ValueError("User events list is empty")

    if observation_date is None:
        observation_date = datetime.utcnow()

    # Sort events by time
    sorted_events = sorted(user_events, key=lambda e: e.occurred_at)

    # ------------------------------------------------------------------
    # Existing 5 features (semantics unchanged)
    # ------------------------------------------------------------------

    # songs_played_total → feature_used events
    songs_played_total = len(
        [e for e in sorted_events if e.event_type == "feature_used"]
    )

    # thumbs_up_count → support_ticket with positive sentiment
    thumbs_up_count = 0
    for event in sorted_events:
        if (
            event.event_type == "support_ticket"
            and event.event_metadata
            and event.event_metadata.get("sentiment") == "positive"
        ):
            thumbs_up_count += 1

    # thumbs_down_count → support_ticket with negative sentiment
    thumbs_down_count = 0
    for event in sorted_events:
        if (
            event.event_type == "support_ticket"
            and event.event_metadata
            and event.event_metadata.get("sentiment") == "negative"
        ):
            thumbs_down_count += 1

    # add_to_playlist_count → feature_used events with playlist feature_name
    add_to_playlist_count = 0
    for event in sorted_events:
        if (
            event.event_type == "feature_used"
            and event.event_metadata
            and event.event_metadata.get("feature_name") == "playlist"
        ):
            add_to_playlist_count += 1

    # avg_session_duration_min — discrete-event approximation:
    # span / event_count converted to minutes-per-event
    time_span_days = (
        sorted_events[-1].occurred_at - sorted_events[0].occurred_at
    ).days
    if time_span_days > 0 and len(sorted_events) > 0:
        avg_session_duration_min = (time_span_days * 24 * 60) / len(sorted_events)
    else:
        avg_session_duration_min = 0.0

    # ------------------------------------------------------------------
    # New features
    # ------------------------------------------------------------------

    # days_since_last_activity — recency vs observation_date
    last_event_at = sorted_events[-1].occurred_at
    # Tolerate naive vs aware datetimes by stripping tz info on both sides
    if last_event_at.tzinfo is not None:
        last_event_at = last_event_at.replace(tzinfo=None)
    obs_date = observation_date
    if obs_date.tzinfo is not None:
        obs_date = obs_date.replace(tzinfo=None)
    delta_days = (obs_date - last_event_at).days
    # If the observation date is before the last event (clock skew, future
    # timestamps, etc.), clamp to 0 rather than emitting a negative value.
    days_since_last_activity = float(max(0, delta_days))

    # session_count_30d — gap-based session clustering in last 30 days.
    # The Postgres `events` table doesn't carry sessionId, so we reconstruct
    # sessions by treating any inter-event gap >30min as a new session.
    cutoff_30d = obs_date - timedelta(days=30)
    recent = [
        e for e in sorted_events
        if (e.occurred_at.replace(tzinfo=None) if e.occurred_at.tzinfo else e.occurred_at)
        >= cutoff_30d
    ]
    if not recent:
        session_count_30d = 0
    else:
        session_count_30d = 1
        prev_ts = recent[0].occurred_at
        if prev_ts.tzinfo is not None:
            prev_ts = prev_ts.replace(tzinfo=None)
        for ev in recent[1:]:
            cur_ts = ev.occurred_at
            if cur_ts.tzinfo is not None:
                cur_ts = cur_ts.replace(tzinfo=None)
            gap_seconds = (cur_ts - prev_ts).total_seconds()
            if gap_seconds > SESSION_GAP_SECONDS:
                session_count_30d += 1
            prev_ts = cur_ts

    # subscription_level — encoding: free=0, anything-paid=1
    plan = (plan_type or "free").lower()
    subscription_level = 0 if plan == "free" else 1

    return {
        "days_since_last_activity": float(days_since_last_activity),
        "session_count_30d": float(session_count_30d),
        "songs_played_total": float(songs_played_total),
        "thumbs_up_count": float(thumbs_up_count),
        "thumbs_down_count": float(thumbs_down_count),
        "add_to_playlist_count": float(add_to_playlist_count),
        "avg_session_duration_min": float(avg_session_duration_min),
        "subscription_level": float(subscription_level),
    }
