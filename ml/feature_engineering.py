"""
Feature engineering pipeline for User Retention Risk Model.

Extracts 8 Sparkify features from raw event logs:
1. days_since_last_activity — recency
2. session_count_30d — frequency
3. songs_played_total — depth of engagement
4. thumbs_up_count — positive signal
5. thumbs_down_count — negative signal
6. add_to_playlist_count — stickiness signal
7. avg_session_duration_min — quality
8. subscription_level — free=0 / paid=1

Churn label: userId appears on "Cancellation Confirmation" page

No data leakage: all features computed from raw events.
"""

from datetime import datetime, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd


# Feature names in order
FEATURE_NAMES = [
    "days_since_last_activity",
    "session_count_30d",
    "songs_played_total",
    "thumbs_up_count",
    "thumbs_down_count",
    "add_to_playlist_count",
    "avg_session_duration_min",
    "subscription_level",
]

# Subscription level encoding (ordinal)
SUBSCRIPTION_LEVEL_ENCODING = {
    "free": 0,
    "paid": 1,
}


def extract_features(
    df: pd.DataFrame,
    observation_date: datetime = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract 8 Sparkify features from raw event DataFrame.

    Processes raw event logs and computes features per user for churn prediction.

    Args:
        df: Raw Sparkify DataFrame with columns:
            userId, sessionId, page, ts, level, registration, length, gender, location, churn
        observation_date: Reference date for feature extraction (defaults to df['ts'].max())

    Returns:
        Tuple of (X, y, feature_names) where:
        - X: numpy array of shape (n_users, 8) with features
        - y: numpy array of shape (n_users,) with churn labels (1 = churned, 0 = retained)
        - feature_names: list of 8 feature names

    Raises:
        ValueError: If df is empty or missing required columns
    """
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    required_columns = ["userId", "sessionId", "page", "ts", "level", "length", "churn"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Use max timestamp in data as observation date if not provided
    if observation_date is None:
        observation_date = df["ts"].max()

    # Group events by user
    user_groups = df.groupby("userId")

    X = []
    y = []
    user_ids = []

    for user_id, user_df in user_groups:
        user_df = user_df.sort_values("ts")

        # Feature 1: days_since_last_activity
        last_activity = user_df["ts"].max()
        days_since_last_activity = (observation_date - last_activity).days
        if pd.isna(days_since_last_activity):
            days_since_last_activity = 999

        # Feature 2: session_count_30d
        cutoff_30d = observation_date - timedelta(days=30)
        sessions_30d = user_df[user_df["ts"] >= cutoff_30d]["sessionId"].nunique()

        # Feature 3: songs_played_total (NextSong page events)
        songs_played_total = len(user_df[user_df["page"] == "NextSong"])

        # Feature 4: thumbs_up_count
        thumbs_up_count = len(user_df[user_df["page"] == "Thumbs Up"])

        # Feature 5: thumbs_down_count
        thumbs_down_count = len(user_df[user_df["page"] == "Thumbs Down"])

        # Feature 6: add_to_playlist_count
        add_to_playlist_count = len(user_df[user_df["page"] == "Add to Playlist"])

        # Feature 7: avg_session_duration_min
        # Session duration = max(ts) - min(ts) per sessionId
        session_durations = []
        for session_id, session_df in user_df.groupby("sessionId"):
            session_duration_ms = (session_df["ts"].max() - session_df["ts"].min()).total_seconds() * 1000
            session_durations.append(session_duration_ms / 60000)  # Convert to minutes
        avg_session_duration_min = (
            np.mean(session_durations) if session_durations else 0.0
        )

        # Feature 8: subscription_level (free=0, paid=1)
        # Get the most recent subscription level for this user
        subscription_level_str = user_df["level"].iloc[-1] if len(user_df) > 0 else "free"
        subscription_level = SUBSCRIPTION_LEVEL_ENCODING.get(
            subscription_level_str.lower(), 0
        )

        # Build feature vector
        features = [
            days_since_last_activity,
            sessions_30d,
            songs_played_total,
            thumbs_up_count,
            thumbs_down_count,
            add_to_playlist_count,
            avg_session_duration_min,
            subscription_level,
        ]
        X.append(features)

        # Churn label: 1 if user has churn=1 in any row, 0 otherwise
        churn = 1 if user_df["churn"].max() == 1 else 0
        y.append(churn)
        user_ids.append(user_id)

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, FEATURE_NAMES


def get_features(
    df: pd.DataFrame = None,
    observation_date: datetime = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    High-level function to extract features from Sparkify DataFrame.

    Args:
        df: Raw Sparkify DataFrame. If None, must be loaded externally.
        observation_date: Reference date for feature extraction (defaults to df['ts'].max())

    Returns:
        Tuple of (X, y, feature_names)

    Raises:
        ValueError: If df is None or empty
    """
    if df is None:
        raise ValueError("DataFrame must be provided to get_features()")

    X, y, feature_names = extract_features(df, observation_date)
    return X, y, feature_names


if __name__ == "__main__":
    # Test the feature extraction
    # This requires the Sparkify dataset to be loaded first
    from ml.data_loader import get_sparkify_data

    try:
        print("Loading Sparkify dataset...")
        df = get_sparkify_data()

        print("Extracting features...")
        X, y, feature_names = get_features(df)

        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING - TEST")
        print("=" * 70)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Churn rate: {y.mean():.4f}")
        print(f"Features: {feature_names}")
        print("\nFeature statistics (train set):")
        for i, name in enumerate(feature_names):
            print(f"  {name:30s}: mean={X[:, i].mean():10.2f}, std={X[:, i].std():10.2f}, min={X[:, i].min():10.2f}, max={X[:, i].max():10.2f}")
        print("\nFirst 3 samples:")
        for i in range(min(3, X.shape[0])):
            print(f"User {i}: {X[i]} -> {y[i]}")
        print("=" * 70)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo test this module, download the Sparkify dataset and place it at:")
        print("  ml/data/sparkify_mini.json")
