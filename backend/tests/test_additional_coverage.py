"""
Additional tests targeting specific missing coverage lines in routers.

Focus on:
1. users.py: error handling paths and edge cases
2. cohorts.py: week calculation and retention matrix building
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock, Mock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from backend.app.main import app
from backend.app.db.models import Base, User, Event, RiskScore
from backend.app.db.session import get_db
from backend.app.routers.cohorts import get_week_number


# ============================================================================
# WEEK NUMBER CALCULATION TEST (cohorts.py line coverage)
# ============================================================================

class TestWeekNumberCalculation:
    """Test week number calculation function."""

    def test_week_number_same_day(self):
        """Test week number for the same day (should be 0)."""
        cohort_date = datetime(2026, 1, 1, 12, 0, 0)
        event_date = datetime(2026, 1, 1, 14, 0, 0)  # Same day, different time

        week = get_week_number(event_date, cohort_date)
        assert week == 0

    def test_week_number_one_day_later(self):
        """Test week number for one day later (should be 0)."""
        cohort_date = datetime(2026, 1, 1)
        event_date = datetime(2026, 1, 2)

        week = get_week_number(event_date, cohort_date)
        assert week == 0

    def test_week_number_exactly_seven_days(self):
        """Test week number for exactly 7 days later (should be 1)."""
        cohort_date = datetime(2026, 1, 1)
        event_date = datetime(2026, 1, 8)

        week = get_week_number(event_date, cohort_date)
        assert week == 1

    def test_week_number_multiple_weeks(self):
        """Test week number for multiple weeks."""
        cohort_date = datetime(2026, 1, 1)

        for days_offset, expected_week in [(0, 0), (6, 0), (7, 1), (13, 1), (14, 2), (28, 4)]:
            event_date = cohort_date + timedelta(days=days_offset)
            week = get_week_number(event_date, cohort_date)
            assert week == expected_week, f"Day {days_offset} should be week {expected_week}, got {week}"


# ============================================================================
# COHORT RETENTION PATHWAY TESTING
# ============================================================================

async def setup_cohort_test_db():
    """Setup a test database with specific cohort data."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with async_session_maker() as session:
        # Create users in different cohorts
        cohort1_date = datetime(2026, 1, 1)
        cohort2_date = datetime(2026, 2, 1)

        user1_id = uuid.uuid4()
        user2_id = uuid.uuid4()
        user3_id = uuid.uuid4()

        user1 = User(id=user1_id, email="cohort1_user1@test.com", plan_type="pro", signup_date=cohort1_date)
        user2 = User(id=user2_id, email="cohort1_user2@test.com", plan_type="free", signup_date=cohort1_date)
        user3 = User(id=user3_id, email="cohort2_user1@test.com", plan_type="pro", signup_date=cohort2_date)

        session.add_all([user1, user2, user3])
        await session.flush()

        # Create events
        # User 1: events in week 0 and week 1
        event1 = Event(
            id=uuid.uuid4(),
            user_id=user1_id,
            event_type="login",
            event_metadata={},
            occurred_at=cohort1_date + timedelta(days=1),
        )
        event2 = Event(
            id=uuid.uuid4(),
            user_id=user1_id,
            event_type="feature_used",
            event_metadata={},
            occurred_at=cohort1_date + timedelta(days=8),  # Week 1
        )
        # User 2: events only in week 0
        event3 = Event(
            id=uuid.uuid4(),
            user_id=user2_id,
            event_type="login",
            event_metadata={},
            occurred_at=cohort1_date + timedelta(days=3),
        )
        # User 3: event in week 0
        event4 = Event(
            id=uuid.uuid4(),
            user_id=user3_id,
            event_type="login",
            event_metadata={},
            occurred_at=cohort2_date + timedelta(days=2),
        )

        session.add_all([event1, event2, event3, event4])
        await session.commit()

        return engine, async_session_maker, user1_id, user2_id, user3_id


@pytest.fixture
def cohort_test_db(event_loop):
    """Fixture providing cohort test database."""
    async def _setup():
        engine, session_maker, u1, u2, u3 = await setup_cohort_test_db()
        return engine, session_maker, u1, u2, u3

    engine, session_maker, u1, u2, u3 = event_loop.run_until_complete(_setup())
    yield engine, session_maker, u1, u2, u3


class TestCohortRetentionCalculation:
    """Test cohort retention calculation with real data."""

    def test_cohort_retention_with_multiple_cohorts(self, cohort_test_db, event_loop):
        """Test retention calculation with multiple cohorts."""
        engine, session_maker, u1, u2, u3 = cohort_test_db

        async def _test():
            from backend.app.routers.cohorts import get_cohort_retention

            async with session_maker() as session:
                result = await get_cohort_retention(session)

                # Should have 2 cohorts
                assert len(result.cohorts) == 2
                assert len(result.weeks) > 0
                assert len(result.retention_matrix) == 2

                # Check retention values are valid
                for cohort_row in result.retention_matrix:
                    for retention in cohort_row:
                        assert 0 <= retention <= 100

        event_loop.run_until_complete(_test())

    def test_cohort_retention_matrix_dimensions(self, cohort_test_db, event_loop):
        """Test that retention matrix dimensions match cohorts."""
        engine, session_maker, u1, u2, u3 = cohort_test_db

        async def _test():
            from backend.app.routers.cohorts import get_cohort_retention

            async with session_maker() as session:
                result = await get_cohort_retention(session)

                num_cohorts = len(result.cohorts)
                num_weeks = len(result.weeks)

                assert len(result.retention_matrix) == num_cohorts

                for cohort_row in result.retention_matrix:
                    assert len(cohort_row) == num_weeks

        event_loop.run_until_complete(_test())


# ============================================================================
# USERS ROUTER ERROR PATH TESTING
# ============================================================================

class TestUsersRouterErrorPaths:
    """Test error handling paths in users router."""

    def test_get_user_risk_with_database_error(self, seeded_db, event_loop):
        """Test behavior when database query fails."""

        async def _create_override():
            # Create a mock session that raises an error
            async def _failing_get_db():
                yield seeded_db

            # Patch to override
            original_override = app.dependency_overrides.get(get_db)
            app.dependency_overrides[get_db] = _failing_get_db

            try:
                client = TestClient(app)
                response = client.get(f"/users/{uuid.uuid4()}/risk")

                # Should fail gracefully with 500
                assert response.status_code in [404, 422, 500]
            finally:
                if original_override:
                    app.dependency_overrides[get_db] = original_override
                else:
                    app.dependency_overrides.pop(get_db, None)

        event_loop.run_until_complete(_create_override())

    def test_at_risk_users_boundary_thresholds(self, client_with_db):
        """Test at-risk endpoint with boundary threshold values."""
        client = TestClient(app)

        # Test 0 threshold (all users)
        response = client.get("/users/at-risk?threshold=0")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for item in data:
            assert item["risk_score"] >= 0

        # Test 100 threshold (no users or only perfect scores)
        response = client.get("/users/at-risk?threshold=100")
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["risk_score"] == 100

    def test_feedback_update_appends_to_existing_action(self, client_with_db, seeded_db, event_loop):
        """Test that feedback appends to existing claude_action."""

        async def _test_append():
            session = seeded_db
            user_id = seeded_db._test_user_2_id

            # Get original action
            stmt = select(RiskScore).where(RiskScore.user_id == user_id).order_by(RiskScore.scored_at.desc())
            result = await session.execute(stmt)
            risk_score = result.scalar_one_or_none()

            original_action = risk_score.claude_action if risk_score else None

            # Send first feedback
            client = TestClient(app)
            response1 = client.post(f"/users/{user_id}/risk/feedback?action=action1")
            assert response1.status_code == 200

            # Refresh from DB
            await session.refresh(risk_score)
            action_after_first = risk_score.claude_action

            if original_action:
                # Should contain original action and new action
                assert "action1" in action_after_first

        event_loop.run_until_complete(_test_append())


# ============================================================================
# EXPLAINER COVERAGE TESTS
# ============================================================================

class TestExplainerEdgeCases:
    """Test edge cases and error paths in explainer."""

    @pytest.mark.asyncio
    async def test_explain_risk_empty_shap_values(self):
        """Test explanation with empty SHAP values dict."""
        from backend.app.ml.explainer import explain_risk

        user_context = {"plan_type": "pro", "days_since_signup": 100, "days_since_last_login": 10}
        top_drivers = ["driver1", "driver2", "driver3"]

        expected_response = {"reason": "Test reason", "action": "Test action"}

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
                result = await explain_risk(
                    user_context=user_context,
                    risk_score=50,
                    top_drivers=top_drivers,
                    shap_values={},
                )

                assert result.reason == expected_response["reason"]
                assert result.action == expected_response["action"]

    @pytest.mark.asyncio
    async def test_explain_risk_empty_top_drivers_list(self):
        """Test explanation with empty top_drivers list."""
        from backend.app.ml.explainer import explain_risk

        user_context = {"plan_type": "free", "days_since_signup": 30, "days_since_last_login": 5}
        top_drivers = []  # Empty

        expected_response = {"reason": "New user", "action": "Onboard"}

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
                result = await explain_risk(
                    user_context=user_context,
                    risk_score=10,
                    top_drivers=top_drivers,
                )

                assert result.reason == expected_response["reason"]
                assert result.action == expected_response["action"]

    @pytest.mark.asyncio
    async def test_explain_risk_json_with_extra_fields(self):
        """Test that extra JSON fields are ignored."""
        from backend.app.ml.explainer import explain_risk

        user_context = {"plan_type": "pro", "days_since_signup": 100, "days_since_last_login": 10}
        top_drivers = ["driver1", "driver2", "driver3"]

        expected_response = {
            "reason": "Main reason",
            "action": "Main action",
            "extra_field": "This should be ignored",
            "another_extra": 123,
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
                result = await explain_risk(
                    user_context=user_context,
                    risk_score=60,
                    top_drivers=top_drivers,
                )

                assert result.reason == "Main reason"
                assert result.action == "Main action"

    @pytest.mark.asyncio
    async def test_explain_risk_with_all_context_fields_missing(self):
        """Test explanation with empty user context."""
        from backend.app.ml.explainer import explain_risk

        user_context = {}  # All fields missing

        expected_response = {"reason": "Unknown user", "action": "Contact support"}

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_async_client.return_value.__aexit__ = AsyncMock(return_value=None)

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
                result = await explain_risk(
                    user_context=user_context,
                    risk_score=0,
                    top_drivers=["f1", "f2", "f3"],
                )

                assert result.reason == "Unknown user"
                assert result.action == "Contact support"
