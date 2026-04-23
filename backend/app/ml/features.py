"""
Feature engineering utilities for live user data (PostgreSQL events).

Wraps ml/feature_engineering.py to extract features for a specific user
from the live event database.
"""

from datetime import datetime, timedelta
from typing import Dict
import numpy as np


def extract_user_features(
    user_events: list,
    observation_date: datetime = None,
) -> Dict[str, float]:
    """
    Extract feature dict for a single user from their events.

    Args:
        user_events: List of event dicts with keys:
            - event_type: str (login, feature_used, support_ticket)
            - event_metadata: dict (optional)
            - occurred_at: datetime
        observation_date: Reference date for feature extraction (defaults to now)

    Returns:
        Dict of {feature_name: value} with all required features for the model

    Model expects (from feature_names.json):
        - songs_played_total
        - thumbs_down_count
        - thumbs_up_count
        - add_to_playlist_count
        - avg_session_duration_min

    Raises:
        ValueError: If user_events is empty
    """
    if not user_events:
        raise ValueError("User events list is empty")

    if observation_date is None:
        observation_date = datetime.utcnow()

    # Sort events by time
    sorted_events = sorted(user_events, key=lambda e: e.occurred_at)

    # Feature 1: songs_played_total → feature_used events (depth)
    # Map: feature_used → songs_played approximation
    songs_played_total = len([e for e in sorted_events if e.event_type == "feature_used"])

    # Feature 2: thumbs_up_count → support_ticket with positive sentiment
    # (from metadata sentiment field if available)
    thumbs_up_count = 0
    for event in sorted_events:
        if (
            event.event_type == "support_ticket"
            and event.event_metadata
            and event.event_metadata.get("sentiment") == "positive"
        ):
            thumbs_up_count += 1

    # Feature 3: thumbs_down_count → support_ticket with negative sentiment
    thumbs_down_count = 0
    for event in sorted_events:
        if (
            event.event_type == "support_ticket"
            and event.event_metadata
            and event.event_metadata.get("sentiment") == "negative"
        ):
            thumbs_down_count += 1

    # Feature 4: add_to_playlist_count → feature_used events with playlist action
    add_to_playlist_count = 0
    for event in sorted_events:
        if (
            event.event_type == "feature_used"
            and event.event_metadata
            and event.event_metadata.get("feature_name") == "playlist"
        ):
            add_to_playlist_count += 1

    # Feature 5: avg_session_duration_min (quality)
    # For our discrete event system, approximate as time span / event count
    time_span_days = (sorted_events[-1].occurred_at - sorted_events[0].occurred_at).days
    if time_span_days > 0 and len(sorted_events) > 0:
        avg_session_duration_min = (time_span_days * 24 * 60) / len(sorted_events)
    else:
        avg_session_duration_min = 0.0

    return {
        "songs_played_total": float(songs_played_total),
        "thumbs_down_count": float(thumbs_down_count),
        "thumbs_up_count": float(thumbs_up_count),
        "add_to_playlist_count": float(add_to_playlist_count),
        "avg_session_duration_min": float(avg_session_duration_min),
    }
