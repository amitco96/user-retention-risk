"""
Tests for the internal /pipeline/* endpoints (Phase 7 nightly job).

Coverage:
- bearer-token auth (missing / wrong / correct / unset env var)
- POST /pipeline/rescore-all response shape and counts
- GET  /pipeline/critical-users returns only critical-tier users
- ML stack is mocked so tests don't depend on live xgboost artifacts
"""

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from backend.app.db.models import RiskScore, User
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.ml.model import RiskPrediction


PIPELINE_SECRET = "test-pipeline-secret-xyz"


@pytest.fixture()
def pipeline_env(monkeypatch):
    """Set API_PIPELINE_SECRET for the duration of the test."""
    monkeypatch.setenv("API_PIPELINE_SECRET", PIPELINE_SECRET)
    yield PIPELINE_SECRET


@pytest.fixture()
def mock_pipeline_ml():
    """
    Patch the ML stack used by the rescore router so tests don't load xgboost.
    Returns a high-tier prediction (risk_score=80) by default.
    """
    pred = RiskPrediction(
        user_id="mock",
        risk_score=80,
        risk_tier="high",
        top_drivers=["thumbs_down_count", "days_since_last_activity"],
        shap_values={"thumbs_down_count": 0.5, "days_since_last_activity": 0.3},
        model_version="1.0",
    )
    with patch("backend.app.routers.pipeline.predict", return_value=pred) as mp, \
         patch("backend.app.routers.pipeline.extract_user_features",
               return_value={"songs_played_total": 10.0}) as mf:
        yield mp, mf


@pytest.fixture()
def pipeline_client(seeded_db, override_get_db, pipeline_env):
    """TestClient wired to the seeded SQLite DB with API_PIPELINE_SECRET set."""
    yield TestClient(app)


def _add_critical_score(db_url: str, user_id: uuid.UUID) -> None:
    """Insert a critical RiskScore row directly into the seeded SQLite db."""
    async def _run():
        engine = create_async_engine(db_url, echo=False, future=True, poolclass=NullPool)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            session.add(RiskScore(
                id=uuid.uuid4(), user_id=user_id, risk_score=95, risk_tier="critical",
                top_drivers=json.dumps(["thumbs_down_count"]),
                shap_values={"thumbs_down_count": 0.9},
                claude_reason="Inactive 21 days; sharp drop in engagement",
                claude_action="Escalate to CSM lead immediately",
                model_version="1.0", scored_at=datetime.utcnow(),
            ))
            await session.commit()
        await engine.dispose()
    asyncio.run(_run())


def _count_rows(db_url: str, stmt_factory):
    async def _run():
        engine = create_async_engine(db_url, echo=False, future=True, poolclass=NullPool)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            result = await session.execute(stmt_factory())
            rows = result.scalars().all()
        await engine.dispose()
        return rows
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestPipelineAuth:
    """Bearer-token gate on both /pipeline/* endpoints."""

    def test_rescore_no_token_returns_401(self, pipeline_client):
        response = pipeline_client.post("/pipeline/rescore-all")
        assert response.status_code == 401

    def test_rescore_wrong_token_returns_401(self, pipeline_client):
        response = pipeline_client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_rescore_malformed_header_returns_401(self, pipeline_client):
        response = pipeline_client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": PIPELINE_SECRET},  # missing "Bearer "
        )
        assert response.status_code == 401

    def test_critical_users_no_token_returns_401(self, pipeline_client):
        response = pipeline_client.get("/pipeline/critical-users")
        assert response.status_code == 401

    def test_critical_users_wrong_token_returns_401(self, pipeline_client):
        response = pipeline_client.get(
            "/pipeline/critical-users",
            headers={"Authorization": "Bearer nope"},
        )
        assert response.status_code == 401

    def test_unset_secret_fails_closed(self, seeded_db, override_get_db, monkeypatch):
        """If API_PIPELINE_SECRET is unset, every request is rejected."""
        monkeypatch.delenv("API_PIPELINE_SECRET", raising=False)
        client = TestClient(app)
        # Even with a bearer header, unset secret = 401
        response = client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": "Bearer anything"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /pipeline/rescore-all
# ---------------------------------------------------------------------------

class TestRescoreAll:
    """Rescore endpoint scores every user and replaces their risk_scores row."""

    def test_rescore_returns_summary_shape(self, pipeline_client, mock_pipeline_ml):
        response = pipeline_client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200
        body = response.json()

        for key in ("scored", "critical", "high", "medium", "low", "duration_seconds"):
            assert key in body, f"missing key: {key}"

        assert isinstance(body["scored"], int)
        assert isinstance(body["duration_seconds"], (int, float))
        assert body["duration_seconds"] >= 0

        # Tier counts must sum to scored.
        tier_total = body["critical"] + body["high"] + body["medium"] + body["low"]
        assert tier_total == body["scored"]

    def test_rescore_scores_all_users_with_events(
        self, pipeline_client, mock_pipeline_ml, seeded_db,
    ):
        """seeded_db has 3 users with events → all 3 should be scored."""
        response = pipeline_client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200
        body = response.json()
        # mock predict always returns risk_tier="high"
        assert body["scored"] == 3
        assert body["high"] == 3
        assert body["critical"] == 0
        assert body["medium"] == 0
        assert body["low"] == 0

    def test_rescore_replaces_existing_rows(
        self, pipeline_client, mock_pipeline_ml, seeded_db,
    ):
        """
        seeded_db already has 2 RiskScore rows. After rescore each user should
        have exactly one row (delete-then-insert upsert semantics).
        """
        response = pipeline_client.post(
            "/pipeline/rescore-all",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200

        rows = _count_rows(seeded_db._db_url, lambda: select(RiskScore))
        # 3 users, 1 row each
        assert len(rows) == 3
        seen_user_ids = {r.user_id for r in rows}
        assert seeded_db._test_user_1_id in seen_user_ids
        assert seeded_db._test_user_2_id in seen_user_ids
        assert seeded_db._test_user_3_id in seen_user_ids


# ---------------------------------------------------------------------------
# GET /pipeline/critical-users
# ---------------------------------------------------------------------------

class TestCriticalUsers:
    """critical-users endpoint returns only critical tier rows."""

    def test_no_critical_users_returns_empty_list(self, pipeline_client, seeded_db):
        """seeded_db has no critical users (only low + high)."""
        response = pipeline_client.get(
            "/pipeline/critical-users",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {"users": []}

    def test_returns_only_critical_tier(self, pipeline_client, seeded_db):
        """Insert a critical row and verify only it comes back."""
        _add_critical_score(seeded_db._db_url, seeded_db._test_user_3_id)

        response = pipeline_client.get(
            "/pipeline/critical-users",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200
        body = response.json()

        assert "users" in body
        assert len(body["users"]) == 1
        u = body["users"][0]
        assert u["user_id"] == str(seeded_db._test_user_3_id)
        assert u["risk_score"] == 95
        assert u["reason"] == "Inactive 21 days; sharp drop in engagement"
        assert u["recommended_action"] == "Escalate to CSM lead immediately"

    def test_response_schema(self, pipeline_client, seeded_db):
        _add_critical_score(seeded_db._db_url, seeded_db._test_user_2_id)

        response = pipeline_client.get(
            "/pipeline/critical-users",
            headers={"Authorization": f"Bearer {PIPELINE_SECRET}"},
        )
        assert response.status_code == 200
        body = response.json()
        for u in body["users"]:
            assert set(u.keys()) == {
                "user_id", "risk_score", "reason", "recommended_action",
            }
            assert isinstance(u["risk_score"], int)
            assert 0 <= u["risk_score"] <= 100
