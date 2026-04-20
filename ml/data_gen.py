#!/usr/bin/env python3
"""
Synthetic data generator for User Retention Risk Model.

Generates realistic user events and inserts into PostgreSQL via SQLAlchemy.

Usage:
    python ml/data_gen.py --users 10000 --seed 42
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import List

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.models import User, Event, RiskScore
from backend.app.db.session import engine, async_session_maker, create_tables


class SyntheticDataGenerator:
    """Generates realistic synthetic user data."""

    # Churn definition: days_since_last_login >= 30 AND sessions_60d < 3
    CHURN_DAYS_THRESHOLD = 30
    CHURN_SESSIONS_60D_THRESHOLD = 3

    def __init__(self, num_users: int, seed: int):
        """
        Initialize the generator.

        Args:
            num_users: Number of users to generate
            seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.seed = seed
        np.random.seed(seed)

        # Set distribution for user cohorts
        self.healthy_ratio = 0.60  # 60% healthy
        self.at_risk_ratio = 0.20  # 20% at-risk
        self.churned_ratio = 0.20  # 20% churned

        # Plan type distribution
        self.plan_distribution = {
            "free": 0.50,
            "starter": 0.30,
            "pro": 0.15,
            "enterprise": 0.05,
        }

    def generate_signup_date(self) -> datetime:
        """Generate signup date between 2 years ago and 6 months ago."""
        days_back = np.random.randint(180, 730)  # 6 months to 2 years ago
        return datetime.utcnow() - timedelta(days=days_back)

    def generate_plan_type(self) -> str:
        """Generate plan type based on distribution."""
        return np.random.choice(
            list(self.plan_distribution.keys()),
            p=list(self.plan_distribution.values()),
        )

    def assign_cohort(self) -> str:
        """Assign user to cohort (healthy, at_risk, churned)."""
        return np.random.choice(
            ["healthy", "at_risk", "churned"],
            p=[self.healthy_ratio, self.at_risk_ratio, self.churned_ratio],
        )

    def generate_events_for_user(
        self, user_id: uuid.UUID, signup_date: datetime, cohort: str
    ) -> List[dict]:
        """
        Generate events for a user based on cohort.

        Args:
            user_id: User ID
            signup_date: User signup date
            cohort: User cohort (healthy, at_risk, churned)

        Returns:
            List of event dicts
        """
        events = []
        now = datetime.utcnow()
        observation_window_start = now - timedelta(days=90)

        if cohort == "healthy":
            # Healthy users: login every 2-3 days, 5-15 feature_used events per week, 0-1 support tickets
            last_login = now - timedelta(
                days=np.random.randint(2, 3)
            )  # Logged in 2-3 days ago
            login_interval = np.random.randint(2, 3)  # Login every 2-3 days

            # Generate logins over the 90-day window
            current_date = max(observation_window_start, signup_date)
            while current_date < now:
                current_date += timedelta(days=login_interval)
                if current_date < now:
                    event_time = current_date + timedelta(
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60),
                    )
                    events.append(
                        {
                            "user_id": user_id,
                            "event_type": "login",
                            "event_metadata": {
                                "session_duration_min": np.random.randint(15, 120),
                            },
                            "occurred_at": event_time,
                        }
                    )

            # Feature usage: 5-15 events per week over 90 days
            num_feature_events = np.random.randint(45, 135)  # ~5-15 per week
            for _ in range(num_feature_events):
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, 90),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                if observation_window_start <= event_time < now:
                    events.append(
                        {
                            "user_id": user_id,
                            "event_type": "feature_used",
                            "event_metadata": {
                                "feature_name": np.random.choice(
                                    [
                                        "dashboard",
                                        "reports",
                                        "settings",
                                        "api",
                                        "exports",
                                    ]
                                ),
                            },
                            "occurred_at": event_time,
                        }
                    )

            # Support tickets: 0-1 over 90 days
            if np.random.random() < 0.3:  # 30% chance of support ticket
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, 90),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                events.append(
                    {
                        "user_id": user_id,
                        "event_type": "support_ticket",
                        "event_metadata": {
                            "status": "resolved",
                            "category": np.random.choice(
                                ["bug", "feature_request", "general"]
                            ),
                        },
                        "occurred_at": event_time,
                    }
                )

        elif cohort == "at_risk":
            # At-risk users: login every 10-20 days, 1-3 feature_used events per week, 2-4 open support tickets
            last_login = now - timedelta(
                days=np.random.randint(10, 20)
            )  # Logged in 10-20 days ago
            login_interval = np.random.randint(10, 20)

            # Generate logins
            current_date = max(observation_window_start, signup_date)
            while current_date < now:
                current_date += timedelta(days=login_interval)
                if current_date < now:
                    event_time = current_date + timedelta(
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60),
                    )
                    events.append(
                        {
                            "user_id": user_id,
                            "event_type": "login",
                            "event_metadata": {
                                "session_duration_min": np.random.randint(5, 30),
                            },
                            "occurred_at": event_time,
                        }
                    )

            # Feature usage: 1-3 per week (9-27 over 90 days)
            num_feature_events = np.random.randint(9, 27)
            for _ in range(num_feature_events):
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, 90),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                if observation_window_start <= event_time < now:
                    events.append(
                        {
                            "user_id": user_id,
                            "event_type": "feature_used",
                            "event_metadata": {
                                "feature_name": np.random.choice(
                                    [
                                        "dashboard",
                                        "reports",
                                        "settings",
                                        "api",
                                        "exports",
                                    ]
                                ),
                            },
                            "occurred_at": event_time,
                        }
                    )

            # Support tickets: 2-4 open tickets
            num_support_tickets = np.random.randint(2, 4)
            for _ in range(num_support_tickets):
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, 90),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                events.append(
                    {
                        "user_id": user_id,
                        "event_type": "support_ticket",
                        "event_metadata": {
                            "status": "open",
                            "category": np.random.choice(
                                ["bug", "feature_request", "general"]
                            ),
                        },
                        "occurred_at": event_time,
                    }
                )

        elif cohort == "churned":
            # Churned users: last login 30+ days ago, <3 sessions in last 60 days, 1-3 open support tickets
            last_login_days = np.random.randint(30, 90)
            last_login_date = now - timedelta(days=last_login_days)

            # Generate one login event in the past 30-90 days
            event_time = last_login_date + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
            )
            if event_time >= observation_window_start:
                events.append(
                    {
                        "user_id": user_id,
                        "event_type": "login",
                        "event_metadata": {
                            "session_duration_min": np.random.randint(1, 10),
                        },
                        "occurred_at": event_time,
                    }
                )

            # Very few feature events: 0-2 over the whole period
            num_feature_events = np.random.randint(0, 2)
            for _ in range(num_feature_events):
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, max(1, last_login_days - 30)),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                if observation_window_start <= event_time < now:
                    events.append(
                        {
                            "user_id": user_id,
                            "event_type": "feature_used",
                            "event_metadata": {
                                "feature_name": np.random.choice(
                                    [
                                        "dashboard",
                                        "reports",
                                        "settings",
                                        "api",
                                        "exports",
                                    ]
                                ),
                            },
                            "occurred_at": event_time,
                        }
                    )

            # Support tickets: 1-3 open tickets
            num_support_tickets = np.random.randint(1, 3)
            for _ in range(num_support_tickets):
                event_time = observation_window_start + timedelta(
                    days=np.random.randint(0, 90),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60),
                )
                events.append(
                    {
                        "user_id": user_id,
                        "event_type": "support_ticket",
                        "event_metadata": {
                            "status": "open",
                            "category": np.random.choice(
                                ["bug", "feature_request", "general"]
                            ),
                        },
                        "occurred_at": event_time,
                    }
                )

        return events

    async def generate_and_insert(self):
        """Generate all synthetic data and insert into database."""
        logger.info(f"Starting data generation: {self.num_users} users, seed={self.seed}")

        async with async_session_maker() as session:
            total_events = 0
            churned_count = 0

            # Batch insert for efficiency
            batch_size = 100
            user_batch = []
            event_batch = []

            for i in range(self.num_users):
                if (i + 1) % 1000 == 0:
                    logger.info(f"Generating user {i + 1}/{self.num_users}")

                # Generate user
                user_id = uuid.uuid4()
                signup_date = self.generate_signup_date()
                plan_type = self.generate_plan_type()
                cohort = self.assign_cohort()

                user = User(
                    id=user_id,
                    email=f"user_{i}_{self.seed}@synthetic.example.com",
                    plan_type=plan_type,
                    signup_date=signup_date,
                )
                user_batch.append(user)

                # Generate events for this user
                events = self.generate_events_for_user(user_id, signup_date, cohort)
                event_batch.extend(
                    [
                        Event(
                            id=uuid.uuid4(),
                            user_id=event["user_id"],
                            event_type=event["event_type"],
                            event_metadata=event["event_metadata"],
                            occurred_at=event["occurred_at"],
                        )
                        for event in events
                    ]
                )
                total_events += len(events)

                # Track churned users for validation
                if cohort == "churned":
                    churned_count += 1

                # Batch insert when we reach batch size
                if len(user_batch) >= batch_size:
                    session.add_all(user_batch)
                    session.add_all(event_batch)
                    await session.flush()
                    user_batch = []
                    event_batch = []

            # Insert remaining batch
            if user_batch:
                session.add_all(user_batch)
                session.add_all(event_batch)

            await session.commit()

            logger.info(f"✓ Generated {self.num_users} users")
            logger.info(f"✓ Generated {total_events} events")
            churn_rate = (churned_count / self.num_users) * 100
            logger.info(f"✓ Churn rate (estimate): {churn_rate:.2f}%")
            logger.info(f"✓ Data insertion complete")

            return {
                "total_users": self.num_users,
                "total_events": total_events,
                "churn_rate": churn_rate,
            }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic user data for retention model"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10000,
        help="Number of users to generate (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create tables first
    logger.info("Creating database tables...")
    await create_tables()
    logger.info("Database tables created/verified")

    # Generate and insert data
    generator = SyntheticDataGenerator(args.users, args.seed)
    result = await generator.generate_and_insert()

    # Print summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Users:       {result['total_users']:,}")
    print(f"Total Events:      {result['total_events']:,}")
    print(f"Churn Rate:        {result['churn_rate']:.2f}%")
    print(f"Avg Events/User:   {result['total_events'] / result['total_users']:.2f}")
    print("=" * 60 + "\n")

    logger.info("Data generation complete!")


if __name__ == "__main__":
    asyncio.run(main())
