"""
Sparkify dataset loader for User Retention Risk Model.

Loads and cleans Sparkify event logs (Udacity medium dataset, ~232MB JSONL).
Parses timestamps, labels churn, and returns clean pandas DataFrame.

Usage:
    from ml.data_loader import get_sparkify_data
    df = get_sparkify_data()
    print(f"Loaded {len(df)} events from {df['userId'].nunique()} users")

Example DataFrame columns after loading:
    userId: str — user identifier
    sessionId: int — session identifier
    page: str — event type (NextSong, Thumbs Up, Add to Playlist, etc.)
    ts: datetime — event timestamp (converted from milliseconds)
    level: str — subscription tier (free/paid)
    registration: datetime — account creation timestamp
    length: float — song duration in seconds
    gender: str — user gender (M/F)
    location: str — user location
    churn: int — 1 if user visited "Cancellation Confirmation", 0 otherwise
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_sparkify_data(
    data_path: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and clean Sparkify dataset.

    Performs the following steps:
    1. Loads JSON event logs from disk
    2. Drops rows with empty/null userId
    3. Converts ts column from milliseconds to datetime
    4. Converts registration column from milliseconds to datetime
    5. Labels churn: 1 if userId visited "Cancellation Confirmation" page, else 0
    6. Cleans nulls in required feature columns
    7. Returns clean DataFrame sorted by userId and ts

    Args:
        data_path: Path to Sparkify JSONL file.
                   Defaults to ml/data/medium-sparkify-event-data.json
        seed: Random seed (for reproducibility, though not used in deterministic loading)

    Returns:
        Clean pandas DataFrame with columns:
        - userId, sessionId, page, ts, level, registration, length, gender, location, churn

    Raises:
        FileNotFoundError: If data_path does not exist
        ValueError: If DataFrame is empty after cleaning
    """
    np.random.seed(seed)

    # Resolve data path
    if data_path is None:
        # Prefer medium dataset; fall back to mini if medium not present
        default_medium = Path(__file__).parent / "data" / "medium-sparkify-event-data.json"
        default_mini = Path(__file__).parent / "data" / "sparkify_mini.json"
        if default_medium.exists():
            data_path = default_medium
        else:
            data_path = default_mini
    else:
        data_path = Path(data_path)

    logger.info(f"Loading Sparkify dataset from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Sparkify dataset not found at {data_path}. "
            "Download from Udacity/Kaggle and place at ml/data/medium-sparkify-event-data.json"
        )

    # Load JSON file
    # The Sparkify dataset is stored as newline-delimited JSON
    df = pd.read_json(data_path, lines=True)
    logger.info(f"Loaded {len(df)} rows from disk")

    # Step 1: Drop rows with empty/null userId
    initial_rows = len(df)
    df = df[df["userId"].notna() & (df["userId"] != "")]
    dropped_userId = initial_rows - len(df)
    if dropped_userId > 0:
        logger.info(f"Dropped {dropped_userId} rows with null/empty userId")

    if len(df) == 0:
        raise ValueError("DataFrame is empty after dropping null userIds")

    # Step 2: Convert ts from milliseconds to datetime
    if "ts" in df.columns and df["ts"].dtype in ["int64", "float64"]:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        logger.info("Converted ts column from milliseconds to datetime")

    # Step 3: Convert registration from milliseconds to datetime
    if "registration" in df.columns and df["registration"].dtype in ["int64", "float64"]:
        df["registration"] = pd.to_datetime(df["registration"], unit="ms", utc=True)
        logger.info("Converted registration column from milliseconds to datetime")

    # Step 4: Label churn: 1 if userId visited "Cancellation Confirmation" page
    churned_users = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()
    df["churn"] = df["userId"].isin(churned_users).astype(int)
    churn_count = df[df["churn"] == 1]["userId"].nunique()
    total_users = df["userId"].nunique()
    churn_rate = (churn_count / total_users) * 100 if total_users > 0 else 0
    logger.info(
        f"Labeled churn: {churn_count}/{total_users} users churned ({churn_rate:.2f}%)"
    )

    # Step 5: Clean nulls in required feature columns
    # Required columns for feature engineering
    required_columns = ["userId", "sessionId", "page", "ts", "level"]
    for col in required_columns:
        if col in df.columns:
            initial_rows = len(df)
            df = df[df[col].notna()]
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with null {col}")

    # Optional columns: fill nulls with defaults
    optional_column_defaults = {
        "registration": pd.NaT,
        "length": 0.0,
        "gender": "U",  # Unknown
        "location": "Unknown",
    }
    for col, default in optional_column_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Step 6: Sort by userId and ts for consistency
    df = df.sort_values(["userId", "ts"]).reset_index(drop=True)

    logger.info(f"Data cleaning complete: {len(df)} events from {df['userId'].nunique()} users")

    return df


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        df = get_sparkify_data()
        print("\n" + "=" * 70)
        print("SPARKIFY DATA LOADER - TEST")
        print("=" * 70)
        print(f"Rows:        {len(df):,}")
        print(f"Users:       {df['userId'].nunique():,}")
        print(f"Sessions:    {df['sessionId'].nunique():,}")
        print(f"Date range:  {df['ts'].min()} to {df['ts'].max()}")
        print(f"Churn rate:  {df['churn'].mean():.4f}")
        print(f"Columns:     {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("=" * 70)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo use this module, download the Sparkify dataset:")
        print("1. Download 'sparkify_mini.json' from Kaggle or Udacity")
        print("2. Create directory: mkdir -p ml/data")
        print("3. Place file at: ml/data/sparkify_mini.json")
