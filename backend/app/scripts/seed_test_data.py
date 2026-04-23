"""
Seed test data into PostgreSQL for API testing.

Usage:
    python -m backend.app.scripts.seed_test_data
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.db.session import async_session_maker
from backend.app.db.models import User, Event


async def seed_data():
    """Create 5 test users with events."""
    async with async_session_maker() as session:
        # Create 5 test users
        users = []
        for i in range(5):
            user = User(
                id=uuid.uuid4(),
                email=f"user{i}@example.com",
                plan_type="paid" if i % 2 == 0 else "free",
                signup_date=datetime.utcnow() - timedelta(days=60 + i*10),
            )
            users.append(user)
            session.add(user)

        await session.flush()

        # Create events for each user
        for user_idx, user in enumerate(users):
            # Generate 10-50 events per user
            num_events = 10 + (user_idx * 5)
            for event_idx in range(num_events):
                event_types = ["login", "feature_used", "support_ticket"]
                event_type = event_types[event_idx % 3]

                # Create event metadata based on event type
                metadata = {}
                if event_type == "feature_used":
                    # Some events are playlist-related
                    metadata["feature_name"] = "playlist" if event_idx % 3 == 0 else "search"
                elif event_type == "support_ticket":
                    # Some support tickets are positive, some negative
                    metadata["sentiment"] = "positive" if event_idx % 2 == 0 else "negative"

                event = Event(
                    id=uuid.uuid4(),
                    user_id=user.id,
                    event_type=event_type,
                    event_metadata=metadata,
                    occurred_at=user.signup_date + timedelta(days=event_idx),
                )
                session.add(event)

        await session.commit()
        print(f"✓ Seeded {len(users)} users with {sum(10 + i*5 for i in range(5))} events")


if __name__ == "__main__":
    asyncio.run(seed_data())
