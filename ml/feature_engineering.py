"""
Feature engineering pipeline for User Retention Risk Model.

Extracts features from PostgreSQL using SQLAlchemy ORM without data leakage.
Features are computed based on data available BEFORE the label period.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.models import User, Event
from backend.app.db.session import async_session_maker


# Feature names in order
FEATURE_NAMES = [
    "days_since_last_login",
    "session_count_30d",
    "feature_usage_count",
    "support_tickets_open",
    "plan_type_encoded",
    "avg_session_duration_min",
    "days_since_signup",
    "login_streak_broken",
]

# Plan type encoding (ordinal)
PLAN_TYPE_ENCODING = {
    "free": 0,
    "starter": 1,
    "pro": 2,
    "enterprise": 3,
}


async def extract_features(
    session: AsyncSession,
    observation_date: datetime,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features for all users from the database.

    Args:
        session: AsyncSession from SQLAlchemy
        observation_date: Reference date for feature extraction (typically now)

    Returns:
        Tuple of (X, y, feature_names) where:
        - X: numpy array of shape (n_users, 8) with features
        - y: numpy array of shape (n_users,) with churn labels
        - feature_names: list of 8 feature names
    """

    # Fetch all users
    user_result = await session.execute(select(User))
    users = user_result.scalars().all()
    user_ids = [user.id for user in users]
    plan_types = {user.id: user.plan_type for user in users}
    signup_dates = {user.id: user.signup_date for user in users}

    # Fetch all events
    event_result = await session.execute(select(Event))
    all_events = event_result.scalars().all()

    # Group events by user
    user_events = {}
    for user_id in user_ids:
        user_events[user_id] = [e for e in all_events if e.user_id == user_id]

    # Extract features for each user
    X = []
    y = []

    for user_id in user_ids:
        events = user_events[user_id]

        # Feature 1: days_since_last_login
        login_events = [e for e in events if e.event_type == "login"]
        if login_events:
            last_login = max(e.occurred_at for e in login_events)
            days_since_last_login = (observation_date - last_login).days
        else:
            # No login events, set to a large number (e.g., 999)
            days_since_last_login = 999

        # Feature 2: session_count_30d (count of login events in last 30 days)
        cutoff_30d = observation_date - timedelta(days=30)
        session_count_30d = sum(1 for e in login_events if e.occurred_at >= cutoff_30d)

        # Feature 3: feature_usage_count (total feature_used events)
        feature_events = [e for e in events if e.event_type == "feature_used"]
        feature_usage_count = len(feature_events)

        # Feature 4: support_tickets_open (count of open support tickets)
        support_events = [e for e in events if e.event_type == "support_ticket"]
        support_tickets_open = sum(
            1
            for e in support_events
            if e.event_metadata and e.event_metadata.get("status") == "open"
        )

        # Feature 5: plan_type_encoded
        plan_type = plan_types.get(user_id, "free")
        plan_type_encoded = PLAN_TYPE_ENCODING.get(plan_type, 0)

        # Feature 6: avg_session_duration_min
        session_durations = []
        for e in login_events:
            if e.event_metadata and "session_duration_min" in e.event_metadata:
                session_durations.append(e.event_metadata["session_duration_min"])
        avg_session_duration_min = (
            np.mean(session_durations) if session_durations else 0.0
        )

        # Feature 7: days_since_signup
        signup_date = signup_dates.get(user_id)
        if signup_date:
            days_since_signup = (observation_date - signup_date).days
        else:
            days_since_signup = 0

        # Feature 8: login_streak_broken (binary: has login broken in last 7 days)
        # This is a binary feature: 1 if there's a gap > 1 day between logins in last 7 days
        cutoff_7d = observation_date - timedelta(days=7)
        recent_logins = sorted(
            [e.occurred_at for e in login_events if e.occurred_at >= cutoff_7d]
        )
        login_streak_broken = 0
        if len(recent_logins) >= 2:
            for i in range(len(recent_logins) - 1):
                gap_days = (recent_logins[i + 1] - recent_logins[i]).days
                if gap_days > 1:
                    login_streak_broken = 1
                    break

        # Build feature vector
        features = [
            days_since_last_login,
            session_count_30d,
            feature_usage_count,
            support_tickets_open,
            plan_type_encoded,
            avg_session_duration_min,
            days_since_signup,
            login_streak_broken,
        ]
        X.append(features)

        # Compute churn label: days_since_last_login >= 30 AND sessions_60d < 3
        cutoff_60d = observation_date - timedelta(days=60)
        sessions_60d = sum(1 for e in login_events if e.occurred_at >= cutoff_60d)
        churn = 1 if (days_since_last_login >= 30 and sessions_60d < 3) else 0
        y.append(churn)

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, FEATURE_NAMES


async def get_features(
    observation_date: datetime = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    High-level function to fetch features using an async session.

    Args:
        observation_date: Reference date for feature extraction (defaults to now)

    Returns:
        Tuple of (X, y, feature_names)
    """
    if observation_date is None:
        observation_date = datetime.utcnow()

    async with async_session_maker() as session:
        X, y, feature_names = await extract_features(session, observation_date)
        return X, y, feature_names


if __name__ == "__main__":
    # Test the feature extraction
    async def test():
        X, y, feature_names = await get_features()
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Churn rate: {y.mean():.4f}")
        print(f"Features: {feature_names}")
        print(f"\nFirst 3 samples:")
        for i in range(min(3, X.shape[0])):
            print(f"User {i}: {X[i]} -> {y[i]}")

    asyncio.run(test())
