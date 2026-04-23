"""
Pytest configuration and fixtures for FastAPI integration tests.

Provides:
- In-memory SQLite async test database with production models
- SQLAlchemy engine and async session factory
- Seeded test data (3 users with events, 2 risk scores)
- Mocked ML model (predict function)
- Mocked Claude explainer (explain_risk function)
- FastAPI dependency override for get_db
"""

import pytest
import uuid
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy import event
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.types import String, VARCHAR

from backend.app.db.models import Base, User, Event, RiskScore
from backend.app.db.session import get_db
from backend.app.ml.model import RiskPrediction
from backend.app.ml.explainer import RiskExplanation
from backend.app.main import app


# Patch SQLAlchemy's ARRAY type to work with SQLite by using JSON
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy import JSON

# Override ARRAY rendering for SQLite to use JSON
SQLiteTypeCompiler.visit_ARRAY = lambda self, type_, **kw: self.visit_JSON(JSON(), **kw)


@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async operations in sync tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def seeded_db(event_loop):
    """
    Sync fixture that creates an in-memory SQLite database with seeded data.

    Uses the production models (User, Event, RiskScore) via the test database.

    Creates test data:
    - user_1: 5 events (low risk), no risk score
    - user_2: 10 events with negative sentiment (high risk), with risk score
    - user_3: 2 events (medium risk), with risk score

    Returns session with _test_user_1_id, _test_user_2_id, _test_user_3_id attributes.
    """

    async def _setup():
        # Create in-memory SQLite engine
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # Create all tables using production models
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create session factory
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

        # Create session
        async with async_session_maker() as session:
            # Create 3 test users
            user_1_id = uuid.uuid4()
            user_2_id = uuid.uuid4()
            user_3_id = uuid.uuid4()

            user_1 = User(
                id=user_1_id,
                email="user1@example.com",
                plan_type="pro",
                signup_date=datetime.utcnow() - timedelta(days=30),
            )
            user_2 = User(
                id=user_2_id,
                email="user2@example.com",
                plan_type="free",
                signup_date=datetime.utcnow() - timedelta(days=60),
            )
            user_3 = User(
                id=user_3_id,
                email="user3@example.com",
                plan_type="starter",
                signup_date=datetime.utcnow() - timedelta(days=15),
            )

            session.add_all([user_1, user_2, user_3])
            await session.flush()

            # Create events for user_1 (5 events, low risk)
            now = datetime.utcnow()
            for i in range(5):
                event_obj = Event(
                    id=uuid.uuid4(),
                    user_id=user_1_id,
                    event_type="login",
                    event_metadata={"source": "web"},
                    occurred_at=now - timedelta(days=5 - i),
                )
                session.add(event_obj)

            # Create events for user_2 (10 events with negative sentiment, high risk)
            for i in range(10):
                sentiment = "negative" if i % 2 == 0 else "positive"
                event_obj = Event(
                    id=uuid.uuid4(),
                    user_id=user_2_id,
                    event_type="support_ticket" if i < 5 else "feature_used",
                    event_metadata={"sentiment": sentiment, "feature_name": "playlist" if i >= 5 else None},
                    occurred_at=now - timedelta(days=15 - i),
                )
                session.add(event_obj)

            # Create events for user_3 (2 events, medium risk)
            for i in range(2):
                event_obj = Event(
                    id=uuid.uuid4(),
                    user_id=user_3_id,
                    event_type="login",
                    event_metadata={"source": "mobile"},
                    occurred_at=now - timedelta(days=3 - i),
                )
                session.add(event_obj)

            await session.flush()

            # Create RiskScore records
            # Note: SQLite stores ARRAY as JSON, so we convert to list
            import json
            risk_score_1 = RiskScore(
                id=uuid.uuid4(),
                user_id=user_1_id,
                risk_score=35,
                risk_tier="low",
                top_drivers=json.dumps(["days_since_last_activity", "session_count_30d", "thumbs_down_count"]),
                shap_values={"days_since_last_activity": 0.2, "session_count_30d": 0.1},
                claude_reason="User has regular engagement with the platform.",
                claude_action="Continue monitoring.",
                model_version="1.0",
                scored_at=now,
            )

            risk_score_2 = RiskScore(
                id=uuid.uuid4(),
                user_id=user_2_id,
                risk_score=80,
                risk_tier="high",
                top_drivers=json.dumps(["thumbs_down_count", "days_since_last_activity", "session_count_30d"]),
                shap_values={"thumbs_down_count": 0.5, "days_since_last_activity": 0.3},
                claude_reason="User has shown multiple negative interactions.",
                claude_action="Reach out with a special offer.",
                model_version="1.0",
                scored_at=now,
            )

            session.add_all([risk_score_1, risk_score_2])
            await session.commit()

            # Store IDs for test access
            session._test_user_1_id = user_1_id
            session._test_user_2_id = user_2_id
            session._test_user_3_id = user_3_id
            session._engine = engine
            session._async_session_maker = async_session_maker

            return session

    session = event_loop.run_until_complete(_setup())
    yield session


@pytest.fixture(scope="function")
def mock_ml_model():
    """
    Mock the ML model predict function.

    Returns a fixed RiskPrediction regardless of input.
    """
    with patch("backend.app.routers.users.predict") as mock_predict:
        mock_predict.return_value = RiskPrediction(
            user_id="test_user",
            risk_score=65,
            risk_tier="medium",
            top_drivers=["session_count_30d", "thumbs_down_count", "days_since_last_activity"],
            shap_values={
                "session_count_30d": 0.3,
                "thumbs_down_count": 0.25,
                "days_since_last_activity": 0.2,
            },
            model_version="1.0",
        )
        yield mock_predict


@pytest.fixture(scope="function")
def mock_claude_explainer():
    """
    Mock the Claude explainer function.

    Returns a fixed RiskExplanation regardless of input.
    """
    with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
        mock_explain.return_value = RiskExplanation(
            reason="User has been inactive for 7 days and shows declining engagement",
            action="Schedule a win-back email campaign this week",
        )
        yield mock_explain


@pytest.fixture(scope="function")
def mock_feature_extraction():
    """
    Mock the feature extraction function.

    Returns a fixed feature dictionary regardless of input.
    """
    with patch("backend.app.routers.users.extract_user_features") as mock_extract:
        mock_extract.return_value = {
            "songs_played_total": 25.0,
            "thumbs_down_count": 5.0,
            "thumbs_up_count": 15.0,
            "add_to_playlist_count": 8.0,
            "avg_session_duration_min": 45.0,
        }
        yield mock_extract


@pytest.fixture(scope="function")
def override_get_db(seeded_db):
    """
    Override FastAPI's get_db dependency to use the test database.

    This allows endpoint tests to use the in-memory SQLite database
    instead of the production PostgreSQL database.
    """
    async def _get_test_db():
        yield seeded_db

    app.dependency_overrides[get_db] = _get_test_db
    yield

    # Cleanup: remove override
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client_with_db(seeded_db, mock_ml_model, mock_claude_explainer, mock_feature_extraction, override_get_db):
    """
    Provide a TestClient with seeded database, mocked ML model, and mocked explainer.

    This fixture combines all dependencies needed for full endpoint integration tests.
    """
    from fastapi.testclient import TestClient

    yield TestClient(app)
