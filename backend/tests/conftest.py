"""
Pytest configuration and fixtures for FastAPI integration tests.

Architecture:
- File-based SQLite database seeded once per test function via asyncio.run()
- NullPool engine in override_get_db → fresh aiosqlite connection per request
  (no event-loop binding; TestClient's background thread creates its own connection)
- TestClient (sync) → coverage.py tracks lines correctly on Python 3.11
"""

import pytest
import uuid
import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import StaticPool, NullPool
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy import JSON

from backend.app.db.models import Base, User, Event, RiskScore
from backend.app.db.session import get_db
from backend.app.ml.model import RiskPrediction
from backend.app.ml.explainer import RiskExplanation
from backend.app.main import app

# Patch SQLAlchemy's ARRAY type to render as JSON on SQLite
SQLiteTypeCompiler.visit_ARRAY = lambda self, type_, **kw: self.visit_JSON(JSON(), **kw)


@pytest.fixture()
def seeded_db():
    """
    Sync fixture: uses asyncio.run() to seed a file-based SQLite database.

    Returns a SimpleNamespace with:
    - _db_url       : sqlite+aiosqlite path to the temp file
    - _test_user_1_id : uuid of user_1 (5 events, risk_score=35)
    - _test_user_2_id : uuid of user_2 (10 events, risk_score=80)
    - _test_user_3_id : uuid of user_3 (2 events, no risk score)
    """
    import json

    db_file = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite+aiosqlite:///{db_file}"

    user_1_id = uuid.uuid4()
    user_2_id = uuid.uuid4()
    user_3_id = uuid.uuid4()

    async def _setup():
        engine = create_async_engine(db_url, echo=False, future=True)
        session_maker = async_sessionmaker(
            engine, class_=AsyncSession,
            expire_on_commit=False, autoflush=False, autocommit=False,
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        now = datetime.utcnow()
        async with session_maker() as session:
            session.add_all([
                User(id=user_1_id, email="user1@example.com", plan_type="pro",
                     signup_date=now - timedelta(days=30)),
                User(id=user_2_id, email="user2@example.com", plan_type="free",
                     signup_date=now - timedelta(days=60)),
                User(id=user_3_id, email="user3@example.com", plan_type="starter",
                     signup_date=now - timedelta(days=15)),
            ])
            await session.flush()

            for i in range(5):
                session.add(Event(id=uuid.uuid4(), user_id=user_1_id, event_type="login",
                                  event_metadata={"source": "web"},
                                  occurred_at=now - timedelta(days=5 - i)))

            for i in range(10):
                session.add(Event(
                    id=uuid.uuid4(), user_id=user_2_id,
                    event_type="support_ticket" if i < 5 else "feature_used",
                    event_metadata={"sentiment": "negative" if i % 2 == 0 else "positive"},
                    occurred_at=now - timedelta(days=15 - i),
                ))

            for i in range(2):
                session.add(Event(id=uuid.uuid4(), user_id=user_3_id, event_type="login",
                                  event_metadata={"source": "mobile"},
                                  occurred_at=now - timedelta(days=3 - i)))

            await session.flush()

            session.add(RiskScore(
                id=uuid.uuid4(), user_id=user_1_id, risk_score=35, risk_tier="low",
                top_drivers=json.dumps(["days_since_last_activity", "session_count_30d"]),
                shap_values={"days_since_last_activity": 0.2, "session_count_30d": 0.1},
                claude_reason="User has regular engagement with the platform.",
                claude_action="Continue monitoring.", model_version="1.0", scored_at=now,
            ))
            session.add(RiskScore(
                id=uuid.uuid4(), user_id=user_2_id, risk_score=80, risk_tier="high",
                top_drivers=json.dumps(["thumbs_down_count", "days_since_last_activity"]),
                shap_values={"thumbs_down_count": 0.5, "days_since_last_activity": 0.3},
                claude_reason="User has shown multiple negative interactions.",
                claude_action="Reach out with a special offer.", model_version="1.0", scored_at=now,
            ))
            await session.commit()

        await engine.dispose()

    asyncio.run(_setup())

    yield SimpleNamespace(
        _db_url=db_url,
        _test_user_1_id=user_1_id,
        _test_user_2_id=user_2_id,
        _test_user_3_id=user_3_id,
    )

    if os.path.exists(db_file):
        os.unlink(db_file)


@pytest.fixture()
def mock_ml_model():
    with patch("backend.app.routers.users.predict") as mock_predict:
        mock_predict.return_value = RiskPrediction(
            user_id="test_user",
            risk_score=65,
            risk_tier="medium",
            top_drivers=["session_count_30d", "thumbs_down_count", "days_since_last_activity"],
            shap_values={"session_count_30d": 0.3, "thumbs_down_count": 0.25,
                         "days_since_last_activity": 0.2},
            model_version="1.0",
        )
        yield mock_predict


@pytest.fixture()
def mock_claude_explainer():
    with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
        mock_explain.return_value = RiskExplanation(
            reason="User has been inactive for 7 days and shows declining engagement",
            action="Schedule a win-back email campaign this week",
        )
        yield mock_explain


@pytest.fixture()
def mock_feature_extraction():
    with patch("backend.app.routers.users.extract_user_features") as mock_extract:
        mock_extract.return_value = {
            "songs_played_total": 25.0,
            "thumbs_down_count": 5.0,
            "thumbs_up_count": 15.0,
            "add_to_playlist_count": 8.0,
            "avg_session_duration_min": 45.0,
        }
        yield mock_extract


@pytest.fixture()
def override_get_db(seeded_db):
    """
    Override FastAPI's get_db dependency.

    Uses NullPool so each request creates a fresh aiosqlite connection in whatever
    event loop is running (TestClient's background thread). No cross-loop binding.
    """
    test_engine = create_async_engine(
        seeded_db._db_url,
        echo=False,
        future=True,
        poolclass=NullPool,
    )
    test_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async def _get_test_db():
        async with test_session_maker() as session:
            yield session

    app.dependency_overrides[get_db] = _get_test_db
    yield test_session_maker

    app.dependency_overrides.clear()
    asyncio.run(test_engine.dispose())


@pytest.fixture()
def client_with_db(seeded_db, mock_ml_model, mock_claude_explainer,
                   mock_feature_extraction, override_get_db):
    """
    Sync fixture: TestClient wired to the seeded database with mocked ML stack.
    Use with synchronous client.get(...) / client.post(...).
    """
    from fastapi.testclient import TestClient
    yield TestClient(app)


@pytest.fixture()
async def async_session(seeded_db):
    """
    Async fixture: yields an AsyncSession bound to the seeded SQLite DB
    in the currently-running event loop.

    Use with direct async function calls (bypassing ASGI/TestClient) so
    coverage.py can track lines after await statements on Python 3.11.
    """
    engine = create_async_engine(
        seeded_db._db_url, echo=False, future=True, poolclass=NullPool,
    )
    session_maker = async_sessionmaker(
        engine, class_=AsyncSession,
        expire_on_commit=False, autoflush=False, autocommit=False,
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest.fixture()
def mock_model():
    """Patch backend.app.routers.users.predict to return a deterministic RiskPrediction."""
    with patch("backend.app.routers.users.predict") as mock_predict:
        mock_predict.return_value = RiskPrediction(
            user_id="test_user",
            risk_score=72,
            risk_tier="high",
            top_drivers=["thumbs_down_count", "days_since_last_activity", "session_count_30d"],
            shap_values={"thumbs_down_count": 0.4, "days_since_last_activity": 0.3,
                         "session_count_30d": 0.2},
            model_version="1.0",
        )
        yield mock_predict


@pytest.fixture()
def mock_explainer():
    """Patch backend.app.routers.users.explain_risk to return a canned RiskExplanation."""
    with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
        mock_explain.return_value = RiskExplanation(
            reason="User engagement has declined sharply in the last 7 days",
            action="Send a targeted win-back email campaign this week",
        )
        yield mock_explain
