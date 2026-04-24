"""
Integration tests for FastAPI routers with database fixtures and mocked dependencies.

Tests:
1. TestGetAtRiskUsers       - GET /users/at-risk?threshold=50
2. TestGetUserRiskById      - GET /users/{user_id}/risk (valid user)
3. TestGetUserRiskNotFound  - GET /users/{user_id}/risk (non-existent user)
4. TestGetUserRiskSavesToDb - Verify RiskScore is saved to database
5. TestPostUserRiskFeedback - POST /users/{user_id}/risk/feedback
6. TestGetCohortsRetention  - GET /cohorts/retention
7. TestGetCohortsRetentionStructure - Verify schema and no NaN values
"""

import asyncio
import uuid
from datetime import datetime

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.app.main import app
from backend.app.db.models import RiskScore
from backend.app.ml.features import extract_user_features


def _query_db(db_url, stmt_factory):
    """Run a single async DB query synchronously via asyncio.run()."""
    async def _run():
        engine = create_async_engine(db_url, echo=False, future=True, poolclass=NullPool)
        session_maker = async_sessionmaker(engine, class_=AsyncSession,
                                           expire_on_commit=False)
        async with session_maker() as session:
            result = await session.execute(stmt_factory())
            rows = result.scalars().all()
        await engine.dispose()
        return rows
    return asyncio.run(_run())


class TestGetAtRiskUsers:
    """Test GET /users/at-risk?threshold endpoint."""

    def test_get_at_risk_users_basic(self, client_with_db):
        """Test that GET /users/at-risk?threshold=50 returns RiskSummary list."""
        response = client_with_db.get("/users/at-risk?threshold=50")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        for risk_summary in data:
            assert risk_summary["risk_score"] >= 50

    def test_get_at_risk_users_high_threshold(self, client_with_db):
        """Test that high threshold returns fewer users."""
        response = client_with_db.get("/users/at-risk?threshold=80")

        assert response.status_code == 200
        data = response.json()
        for risk_summary in data:
            assert risk_summary["risk_score"] >= 80

    def test_get_at_risk_users_response_schema(self, client_with_db):
        """Test that response follows RiskSummary schema."""
        response = client_with_db.get("/users/at-risk?threshold=30")

        assert response.status_code == 200
        data = response.json()
        for risk_summary in data:
            assert "user_id" in risk_summary
            assert "risk_score" in risk_summary
            assert "risk_tier" in risk_summary
            assert "reason" in risk_summary
            assert isinstance(risk_summary["risk_score"], int)
            assert 0 <= risk_summary["risk_score"] <= 100

    def test_get_at_risk_users_sorted_by_score(self, client_with_db):
        """Test that results are sorted by risk_score descending."""
        response = client_with_db.get("/users/at-risk?threshold=0")

        assert response.status_code == 200
        data = response.json()
        if len(data) > 1:
            for i in range(len(data) - 1):
                assert data[i]["risk_score"] >= data[i + 1]["risk_score"]

    def test_get_at_risk_users_default_threshold(self, client_with_db):
        """Test that default threshold=70 works."""
        response = client_with_db.get("/users/at-risk")

        assert response.status_code == 200
        data = response.json()
        for risk_summary in data:
            assert risk_summary["risk_score"] >= 70


class TestGetUserRiskById:
    """Test GET /users/{user_id}/risk endpoint (success path)."""

    def test_get_user_risk_endpoint_returns_200(self, client_with_db, seeded_db):
        """Test that GET /users/{user_id}/risk returns 200 for a known user."""
        user_id = seeded_db._test_user_1_id
        response = client_with_db.get(f"/users/{user_id}/risk")
        assert response.status_code == 200

    def test_get_user_risk_response_schema(self, client_with_db, seeded_db):
        """Test that 200 response matches RiskResponse schema."""
        user_id = seeded_db._test_user_1_id
        response = client_with_db.get(f"/users/{user_id}/risk")

        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "risk_score" in data
        assert "risk_tier" in data
        assert "top_drivers" in data
        assert "reason" in data
        assert "recommended_action" in data
        assert isinstance(data["risk_score"], int)
        assert 0 <= data["risk_score"] <= 100


class TestGetUserRiskNotFound:
    """Test GET /users/{user_id}/risk with non-existent user."""

    def test_get_user_risk_unknown_user_returns_404(self, client_with_db):
        """Test that non-existent user returns 404."""
        fake_id = uuid.uuid4()
        response = client_with_db.get(f"/users/{fake_id}/risk")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_user_risk_invalid_uuid_returns_422(self, client_with_db):
        """Test that invalid UUID format returns 422."""
        for invalid_id in ["not-a-uuid", "123", "user@example.com"]:
            response = client_with_db.get(f"/users/{invalid_id}/risk")
            assert response.status_code == 422


class TestGetUserRiskSavesToDb:
    """Test that GET /users/{user_id}/risk saves to risk_scores table."""

    def test_get_user_risk_saves_to_risk_scores_table(self, client_with_db, seeded_db):
        """Test that a successful GET /users/{user_id}/risk saves a RiskScore record."""
        user_id = seeded_db._test_user_1_id

        response = client_with_db.get(f"/users/{user_id}/risk")
        assert response.status_code == 200

        rows = _query_db(
            seeded_db._db_url,
            lambda: select(RiskScore).where(RiskScore.user_id == user_id)
                                     .order_by(RiskScore.scored_at.desc()),
        )
        assert len(rows) >= 1
        latest = rows[0]
        assert latest.risk_score > 0
        assert latest.risk_tier in ["low", "medium", "high", "critical"]


class TestPostUserRiskFeedback:
    """Test POST /users/{user_id}/risk/feedback endpoint."""

    def test_post_user_risk_feedback_success(self, client_with_db, seeded_db):
        """Test that POST feedback returns 200 with success message."""
        user_id = seeded_db._test_user_2_id

        response = client_with_db.post(
            f"/users/{user_id}/risk/feedback?action=email_sent&notes=Sent+retention+email",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data

    def test_post_user_risk_feedback_updates_db(self, client_with_db, seeded_db):
        """Test that feedback appends the CSM action to the most recent RiskScore."""
        user_id = seeded_db._test_user_2_id

        response = client_with_db.post(
            f"/users/{user_id}/risk/feedback?action=call_scheduled&notes=Scheduled+call",
        )
        assert response.status_code == 200

        rows = _query_db(
            seeded_db._db_url,
            lambda: select(RiskScore).where(RiskScore.user_id == user_id)
                                     .order_by(RiskScore.scored_at.desc()),
        )
        assert len(rows) >= 1
        assert "call_scheduled" in rows[0].claude_action

    def test_post_user_risk_feedback_unknown_user_returns_404(self, client_with_db):
        """Test that feedback for unknown user returns 404."""
        fake_id = uuid.uuid4()
        response = client_with_db.post(f"/users/{fake_id}/risk/feedback?action=email_sent")
        assert response.status_code == 404

    def test_post_user_risk_feedback_invalid_uuid_returns_422(self, client_with_db):
        """Test that invalid UUID format returns 422."""
        response = client_with_db.post("/users/invalid-uuid/risk/feedback?action=email_sent")
        assert response.status_code == 422


class TestCohortHelpers:
    """Test cohort helper functions."""

    def test_get_week_number_same_week(self):
        """Test get_week_number for dates in same week."""
        from backend.app.routers.cohorts import get_week_number

        cohort_start = datetime(2026, 1, 1)
        assert get_week_number(cohort_start, cohort_start) == 0
        assert get_week_number(datetime(2026, 1, 4), cohort_start) == 0

    def test_get_week_number_different_weeks(self):
        """Test get_week_number for dates in different weeks."""
        from backend.app.routers.cohorts import get_week_number

        cohort_start = datetime(2026, 1, 1)
        assert get_week_number(datetime(2026, 1, 8), cohort_start) == 1
        assert get_week_number(datetime(2026, 1, 15), cohort_start) == 2

    def test_get_week_number_before_cohort(self):
        """Test get_week_number with date before cohort start returns negative."""
        from backend.app.routers.cohorts import get_week_number

        cohort_start = datetime(2026, 1, 15)
        assert get_week_number(datetime(2026, 1, 1), cohort_start) < 0


class TestGetCohortsRetention:
    """Test GET /cohorts/retention endpoint."""

    def test_get_cohorts_retention_returns_200(self, client_with_db):
        """Test that GET /cohorts/retention returns 200."""
        response = client_with_db.get("/cohorts/retention")
        assert response.status_code == 200

    def test_get_cohorts_retention_valid_schema(self, client_with_db):
        """Test that response follows CohortRetentionData schema."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        assert "cohorts" in data
        assert "weeks" in data
        assert "retention_matrix" in data
        assert isinstance(data["cohorts"], list)
        assert isinstance(data["weeks"], list)
        assert isinstance(data["retention_matrix"], list)

    def test_get_cohorts_retention_matrix_shape(self, client_with_db):
        """Test that retention_matrix has correct dimensions."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        assert len(data["retention_matrix"]) == len(data["cohorts"])

    def test_get_cohorts_retention_values_in_range(self, client_with_db):
        """Test that retention percentages are between 0-100."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        for cohort_row in data["retention_matrix"]:
            for retention_pct in cohort_row:
                assert 0 <= retention_pct <= 100

    def test_get_cohorts_retention_no_nan_values(self, client_with_db):
        """Test that there are no NaN or infinite values in retention."""
        import math

        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        for cohort_row in data["retention_matrix"]:
            for retention_pct in cohort_row:
                assert not math.isnan(retention_pct)
                assert math.isfinite(retention_pct)

    def test_cohorts_populated_from_seeded_users(self, client_with_db):
        """Test that cohorts are derived from seeded user signup dates."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        assert len(data["cohorts"]) >= 1
        assert len(data["weeks"]) >= 1
        assert len(data["retention_matrix"]) == len(data["cohorts"])
        for cohort_row in data["retention_matrix"]:
            assert len(cohort_row) == len(data["weeks"])


class TestGetCohortsRetentionStructure:
    """Test CohortRetentionData schema structure."""

    def test_get_cohorts_retention_cohort_labels(self, client_with_db):
        """Test that cohort labels are non-empty strings."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        for cohort in data["cohorts"]:
            assert isinstance(cohort, str)
            assert len(cohort) > 0

    def test_get_cohorts_retention_week_labels(self, client_with_db):
        """Test that week labels are non-empty strings."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        for week in data["weeks"]:
            assert isinstance(week, str)
            assert "Week" in week

    def test_get_cohorts_retention_consistency(self, client_with_db):
        """Test that retention_matrix dimensions match cohorts count."""
        response = client_with_db.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()
        assert len(data["retention_matrix"]) == len(data["cohorts"])


class TestGetUserRiskWithRealFeatures:
    """Test GET /users/{user_id}/risk with real feature extraction (mocking only predict + explain)."""

    def test_get_user_risk_with_real_features(self, seeded_db, override_get_db,
                                               mock_ml_model, mock_claude_explainer):
        """Test that endpoint scores a user using real feature extraction."""
        user_id = seeded_db._test_user_1_id

        client = TestClient(app)
        response = client.get(f"/users/{user_id}/risk")

        assert response.status_code == 200
        data = response.json()
        assert data["risk_score"] == 65
        assert data["risk_tier"] == "medium"
        assert isinstance(data["top_drivers"], list)
        assert len(data["top_drivers"]) > 0

    def test_extract_user_features_called_with_events(self, seeded_db, override_get_db,
                                                       mock_ml_model, mock_claude_explainer):
        """Test that endpoint calls extract_user_features with real Event objects."""
        from unittest.mock import patch

        user_id = seeded_db._test_user_2_id

        with patch("backend.app.routers.users.extract_user_features",
                   wraps=extract_user_features) as mock_extract:
            client = TestClient(app)
            response = client.get(f"/users/{user_id}/risk")

            assert response.status_code == 200
            assert mock_extract.called
            call_args = mock_extract.call_args[0][0]
            assert len(call_args) > 0
            assert hasattr(call_args[0], "event_type")

    def test_get_user_risk_saves_risk_score_record(self, seeded_db, override_get_db,
                                                    mock_ml_model, mock_claude_explainer):
        """Test that endpoint saves RiskScore record to database."""
        user_id = seeded_db._test_user_1_id

        client = TestClient(app)
        response = client.get(f"/users/{user_id}/risk")

        assert response.status_code == 200

        rows = _query_db(
            seeded_db._db_url,
            lambda: select(RiskScore).where(RiskScore.user_id == user_id)
                                     .order_by(RiskScore.scored_at.desc())
                                     .limit(1),
        )
        assert len(rows) >= 1
        assert rows[0].risk_score == 65
        assert rows[0].risk_tier == "medium"


class TestRouterLogicDirect:
    """
    Direct async function call tests that bypass ASGI transport.

    Why: coverage.py 7.x on Python 3.11 cannot track lines immediately after
    await statements in async router functions invoked through Starlette's
    ASGI transport. Calling the endpoint coroutines directly restores
    coverage tracking while still exercising the real router logic.
    """

    async def test_get_user_risk_direct(
        self, async_session, seeded_db, mock_model, mock_explainer,
    ):
        from backend.app.routers.users import get_user_risk

        user_id = str(seeded_db._test_user_1_id)
        result = await get_user_risk(user_id=user_id, db=async_session)

        assert result.user_id == user_id
        assert 0 <= result.risk_score <= 100
        assert result.risk_tier in ["low", "medium", "high", "critical"]
        assert isinstance(result.top_drivers, list)
        assert result.reason
        assert result.recommended_action
        assert result.model_version

    async def test_get_user_risk_direct_invalid_uuid_raises_422(
        self, async_session, mock_model, mock_explainer,
    ):
        from backend.app.routers.users import get_user_risk

        with pytest.raises(HTTPException) as excinfo:
            await get_user_risk(user_id="not-a-uuid", db=async_session)
        assert excinfo.value.status_code == 422

    async def test_get_user_risk_direct_unknown_user_raises_404(
        self, async_session, mock_model, mock_explainer,
    ):
        from backend.app.routers.users import get_user_risk

        unknown = str(uuid.uuid4())
        with pytest.raises(HTTPException) as excinfo:
            await get_user_risk(user_id=unknown, db=async_session)
        assert excinfo.value.status_code == 404

    async def test_get_user_risk_direct_user_without_events_raises_422(
        self, async_session, seeded_db, mock_model, mock_explainer,
    ):
        """user_3 has 2 events — need a user with zero events to exercise the 422 branch."""
        from backend.app.db.models import User
        from backend.app.routers.users import get_user_risk

        # Create a user with no events
        new_user_id = uuid.uuid4()
        async_session.add(User(
            id=new_user_id,
            email="noevents@example.com",
            plan_type="free",
            signup_date=datetime.utcnow(),
        ))
        await async_session.commit()

        with pytest.raises(HTTPException) as excinfo:
            await get_user_risk(user_id=str(new_user_id), db=async_session)
        assert excinfo.value.status_code == 422

    async def test_get_at_risk_users_direct(self, async_session):
        from backend.app.routers.users import get_at_risk_users

        result = await get_at_risk_users(threshold=50, db=async_session)
        assert isinstance(result, list)
        for summary in result:
            assert summary.risk_score >= 50
            assert summary.risk_tier in ["low", "medium", "high", "critical"]

    async def test_get_at_risk_users_direct_high_threshold(self, async_session):
        from backend.app.routers.users import get_at_risk_users

        result = await get_at_risk_users(threshold=95, db=async_session)
        assert isinstance(result, list)
        for summary in result:
            assert summary.risk_score >= 95

    async def test_get_at_risk_users_direct_sorted(self, async_session):
        from backend.app.routers.users import get_at_risk_users

        result = await get_at_risk_users(threshold=0, db=async_session)
        assert isinstance(result, list)
        for i in range(len(result) - 1):
            assert result[i].risk_score >= result[i + 1].risk_score

    async def test_record_risk_feedback_direct(self, async_session, seeded_db):
        from backend.app.routers.users import record_risk_feedback

        user_id = str(seeded_db._test_user_2_id)
        result = await record_risk_feedback(
            user_id=user_id,
            action="email_sent",
            notes="Direct-call test",
            db=async_session,
        )
        assert result["status"] == "success"
        assert user_id in result["message"]

    async def test_record_risk_feedback_direct_invalid_uuid(self, async_session):
        from backend.app.routers.users import record_risk_feedback

        with pytest.raises(HTTPException) as excinfo:
            await record_risk_feedback(
                user_id="bad-uuid", action="email_sent", db=async_session,
            )
        assert excinfo.value.status_code == 422

    async def test_record_risk_feedback_direct_unknown_user_404(self, async_session):
        from backend.app.routers.users import record_risk_feedback

        with pytest.raises(HTTPException) as excinfo:
            await record_risk_feedback(
                user_id=str(uuid.uuid4()),
                action="email_sent",
                db=async_session,
            )
        assert excinfo.value.status_code == 404

    async def test_record_risk_feedback_direct_appends_on_existing_action(
        self, async_session, seeded_db,
    ):
        """user_2 has a seeded RiskScore with claude_action already set,
        so the 'append' branch executes."""
        from backend.app.routers.users import record_risk_feedback

        user_id = str(seeded_db._test_user_2_id)
        await record_risk_feedback(
            user_id=user_id, action="call_scheduled", db=async_session,
        )

        stmt = (select(RiskScore)
                .where(RiskScore.user_id == seeded_db._test_user_2_id)
                .order_by(RiskScore.scored_at.desc()))
        rows = (await async_session.execute(stmt)).scalars().all()
        assert "call_scheduled" in rows[0].claude_action

    async def test_get_cohort_retention_direct(self, async_session):
        from backend.app.routers.cohorts import get_cohort_retention

        result = await get_cohort_retention(db=async_session)
        assert hasattr(result, "cohorts")
        assert hasattr(result, "weeks")
        assert hasattr(result, "retention_matrix")
        assert isinstance(result.cohorts, list)
        assert isinstance(result.weeks, list)
        assert isinstance(result.retention_matrix, list)
        assert len(result.retention_matrix) == len(result.cohorts)
        for row in result.retention_matrix:
            for pct in row:
                assert 0 <= pct <= 100

    async def test_get_cohort_retention_direct_no_users(self, async_session):
        """Empty-DB branch: delete all rows, call the endpoint, expect empty matrix."""
        from backend.app.db.models import Event, RiskScore as RS, User
        from backend.app.routers.cohorts import get_cohort_retention

        # Delete in FK order: risk_scores -> events -> users
        await async_session.execute(RS.__table__.delete())
        await async_session.execute(Event.__table__.delete())
        await async_session.execute(User.__table__.delete())
        await async_session.commit()

        result = await get_cohort_retention(db=async_session)
        assert result.cohorts == []
        assert result.weeks == []
        assert result.retention_matrix == []

    async def test_get_cohort_retention_direct_users_without_events(self, async_session):
        """Users exist but no events: returns cohorts with empty per-cohort rows."""
        from backend.app.db.models import Event, RiskScore as RS
        from backend.app.routers.cohorts import get_cohort_retention

        # Keep users, drop events + risk_scores
        await async_session.execute(RS.__table__.delete())
        await async_session.execute(Event.__table__.delete())
        await async_session.commit()

        result = await get_cohort_retention(db=async_session)
        assert len(result.cohorts) >= 1
        assert result.weeks == []
        assert len(result.retention_matrix) == len(result.cohorts)


class TestRouterErrorPaths:
    """
    Direct async tests that force exceptions inside router handlers to cover the
    `except Exception` / `except ValueError` branches and the `else` branch in
    record_risk_feedback.
    """

    async def test_get_user_risk_direct_predict_raises_valueerror(
        self, async_session, seeded_db, mock_explainer,
    ):
        """Force predict() to raise ValueError → triggers the 422 ValueError branch."""
        from unittest.mock import patch
        from backend.app.routers.users import get_user_risk

        user_id = str(seeded_db._test_user_1_id)
        with patch("backend.app.routers.users.predict", side_effect=ValueError("bad features")):
            with pytest.raises(HTTPException) as excinfo:
                await get_user_risk(user_id=user_id, db=async_session)
        assert excinfo.value.status_code == 422

    async def test_get_user_risk_direct_predict_raises_generic(
        self, async_session, seeded_db, mock_explainer,
    ):
        """Force predict() to raise RuntimeError → triggers the 500 generic branch."""
        from unittest.mock import patch
        from backend.app.routers.users import get_user_risk

        user_id = str(seeded_db._test_user_1_id)
        with patch("backend.app.routers.users.predict", side_effect=RuntimeError("boom")):
            with pytest.raises(HTTPException) as excinfo:
                await get_user_risk(user_id=user_id, db=async_session)
        assert excinfo.value.status_code == 500

    async def test_get_at_risk_users_direct_db_error_returns_500(self, async_session):
        """Force db.execute to raise → triggers except Exception branch."""
        from unittest.mock import patch, AsyncMock
        from backend.app.routers.users import get_at_risk_users

        with patch.object(async_session, "execute",
                          new=AsyncMock(side_effect=RuntimeError("db down"))):
            with pytest.raises(HTTPException) as excinfo:
                await get_at_risk_users(threshold=50, db=async_session)
        assert excinfo.value.status_code == 500

    async def test_record_risk_feedback_direct_db_error_returns_500(
        self, async_session, seeded_db,
    ):
        """Force commit to raise after we find the RiskScore → triggers except Exception."""
        from unittest.mock import patch, AsyncMock
        from backend.app.routers.users import record_risk_feedback

        user_id = str(seeded_db._test_user_1_id)
        with patch.object(async_session, "commit",
                          new=AsyncMock(side_effect=RuntimeError("commit failed"))):
            with pytest.raises(HTTPException) as excinfo:
                await record_risk_feedback(
                    user_id=user_id, action="email_sent", db=async_session,
                )
        assert excinfo.value.status_code == 500

    async def test_record_risk_feedback_direct_else_branch_sets_action(
        self, async_session, seeded_db,
    ):
        """RiskScore with claude_action=None → triggers the `else` assignment branch."""
        import uuid as _uuid
        from backend.app.db.models import RiskScore as RS
        from backend.app.routers.users import record_risk_feedback

        # Seed a RiskScore with claude_action = None for a fresh user
        new_user_id = seeded_db._test_user_3_id
        async_session.add(RS(
            id=_uuid.uuid4(),
            user_id=new_user_id,
            risk_score=50, risk_tier="medium",
            top_drivers="[]", shap_values={},
            claude_reason=None, claude_action=None,
            model_version="1.0", scored_at=datetime.utcnow(),
        ))
        await async_session.commit()

        result = await record_risk_feedback(
            user_id=str(new_user_id), action="new_action", db=async_session,
        )
        assert result["status"] == "success"

        stmt = (select(RS).where(RS.user_id == new_user_id)
                          .order_by(RS.scored_at.desc()))
        rows = (await async_session.execute(stmt)).scalars().all()
        # else-branch sets claude_action directly to the action string (no "[CSM:" prefix)
        assert rows[0].claude_action == "new_action"

    async def test_get_cohort_retention_direct_db_error_returns_500(self, async_session):
        """Force db.execute to raise → triggers except Exception branch in cohorts."""
        from unittest.mock import patch, AsyncMock
        from backend.app.routers.cohorts import get_cohort_retention

        with patch.object(async_session, "execute",
                          new=AsyncMock(side_effect=RuntimeError("db down"))):
            with pytest.raises(HTTPException) as excinfo:
                await get_cohort_retention(db=async_session)
        assert excinfo.value.status_code == 500
