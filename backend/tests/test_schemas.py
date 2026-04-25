"""
Unit tests for Pydantic schemas (backend/app/schemas/risk.py).

Tests schema validation and serialization.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from backend.app.schemas.risk import (
    RiskResponse,
    RiskSummary,
    CohortRetentionData,
    Cohort,
    CohortWeekData,
)


class TestRiskResponse:
    """Test RiskResponse schema."""

    def test_risk_response_valid_data(self):
        """Test RiskResponse with valid data."""
        response = RiskResponse(
            user_id="user-123",
            risk_score=75,
            risk_tier="high",
            top_drivers=["feature1", "feature2", "feature3"],
            reason="User engagement is declining.",
            recommended_action="Send re-engagement email.",
            scored_at=datetime.utcnow(),
            model_version="1.0",
        )

        assert response.user_id == "user-123"
        assert response.risk_score == 75
        assert response.risk_tier == "high"
        assert len(response.top_drivers) == 3
        assert response.reason == "User engagement is declining."
        assert response.recommended_action == "Send re-engagement email."
        assert response.model_version == "1.0"

    def test_risk_response_risk_score_bounds(self):
        """Test that risk_score must be 0-100."""
        # Valid
        response = RiskResponse(
            user_id="user-123",
            risk_score=0,
            risk_tier="low",
            top_drivers=["f1", "f2", "f3"],
            reason="Low risk",
            recommended_action="Monitor.",
            scored_at=datetime.utcnow(),
        )
        assert response.risk_score == 0

        response = RiskResponse(
            user_id="user-123",
            risk_score=100,
            risk_tier="critical",
            top_drivers=["f1", "f2", "f3"],
            reason="High risk",
            recommended_action="Intervene.",
            scored_at=datetime.utcnow(),
        )
        assert response.risk_score == 100

        # Invalid
        with pytest.raises(ValidationError):
            RiskResponse(
                user_id="user-123",
                risk_score=101,  # Out of bounds
                risk_tier="critical",
                top_drivers=["f1", "f2", "f3"],
                reason="Invalid",
                recommended_action="Action",
                scored_at=datetime.utcnow(),
            )

    def test_risk_response_risk_tier_validation(self):
        """Test that risk_tier is validated (no validation in schema currently)."""
        response = RiskResponse(
            user_id="user-123",
            risk_score=50,
            risk_tier="medium",
            top_drivers=["f1", "f2", "f3"],
            reason="Medium risk",
            recommended_action="Monitor.",
            scored_at=datetime.utcnow(),
        )
        assert response.risk_tier == "medium"

    def test_risk_response_default_model_version(self):
        """Test that model_version defaults to '1.0'."""
        response = RiskResponse(
            user_id="user-123",
            risk_score=50,
            risk_tier="medium",
            top_drivers=["f1", "f2", "f3"],
            reason="Test",
            recommended_action="Test",
            scored_at=datetime.utcnow(),
        )
        assert response.model_version == "1.0"

    def test_risk_response_serialization(self):
        """Test that RiskResponse can be serialized to JSON."""
        response = RiskResponse(
            user_id="user-123",
            risk_score=75,
            risk_tier="high",
            top_drivers=["f1", "f2", "f3"],
            reason="Test reason",
            recommended_action="Test action",
            scored_at=datetime.utcnow(),
        )

        # Should be serializable to dict/JSON
        data = response.model_dump()
        assert data["user_id"] == "user-123"
        assert data["risk_score"] == 75
        assert data["risk_tier"] == "high"


class TestRiskSummary:
    """Test RiskSummary schema."""

    def test_risk_summary_valid_data(self):
        """Test RiskSummary with valid data."""
        summary = RiskSummary(
            user_id="user-123",
            risk_score=75,
            risk_tier="high",
            reason="High inactivity",
        )

        assert summary.user_id == "user-123"
        assert summary.risk_score == 75
        assert summary.risk_tier == "high"
        assert summary.reason == "High inactivity"

    def test_risk_summary_reason_defaults_to_empty(self):
        """Test that reason defaults to empty string."""
        summary = RiskSummary(
            user_id="user-123",
            risk_score=50,
            risk_tier="medium",
        )

        assert summary.reason == ""

    def test_risk_summary_risk_score_bounds(self):
        """Test that risk_score must be 0-100."""
        # Valid
        summary = RiskSummary(
            user_id="user-123",
            risk_score=0,
            risk_tier="low",
        )
        assert summary.risk_score == 0

        summary = RiskSummary(
            user_id="user-123",
            risk_score=100,
            risk_tier="critical",
        )
        assert summary.risk_score == 100

        # Invalid
        with pytest.raises(ValidationError):
            RiskSummary(
                user_id="user-123",
                risk_score=101,  # Out of bounds
                risk_tier="critical",
            )

    def test_risk_summary_serialization(self):
        """Test that RiskSummary can be serialized."""
        summary = RiskSummary(
            user_id="user-123",
            risk_score=75,
            risk_tier="high",
            reason="Test",
        )

        data = summary.model_dump()
        assert data["user_id"] == "user-123"
        assert data["risk_score"] == 75


class TestCohortRetentionData:
    """Test CohortRetentionData schema (nested cohorts -> weeks shape)."""

    def test_cohort_retention_data_valid(self):
        """Test CohortRetentionData with valid nested data."""
        data = CohortRetentionData(
            cohorts=[
                Cohort(cohort_week="2024-01", weeks=[
                    CohortWeekData(week=0, retention_pct=100.0),
                    CohortWeekData(week=1, retention_pct=85.0),
                    CohortWeekData(week=2, retention_pct=70.0),
                ]),
                Cohort(cohort_week="2024-02", weeks=[
                    CohortWeekData(week=0, retention_pct=98.0),
                    CohortWeekData(week=1, retention_pct=80.0),
                ]),
                Cohort(cohort_week="2024-03", weeks=[
                    CohortWeekData(week=0, retention_pct=95.0),
                ]),
            ],
        )

        assert len(data.cohorts) == 3
        assert data.cohorts[0].cohort_week == "2024-01"
        assert len(data.cohorts[0].weeks) == 3
        assert data.cohorts[0].weeks[1].retention_pct == 85.0

    def test_cohort_retention_empty_data(self):
        """Test CohortRetentionData with empty cohorts list."""
        data = CohortRetentionData(cohorts=[])
        assert data.cohorts == []

    def test_cohort_retention_default_empty(self):
        """Test that CohortRetentionData() defaults cohorts to empty list."""
        data = CohortRetentionData()
        assert data.cohorts == []

    def test_cohort_retention_single_cohort(self):
        """Test CohortRetentionData with single cohort."""
        data = CohortRetentionData(
            cohorts=[
                Cohort(cohort_week="2024-01", weeks=[
                    CohortWeekData(week=0, retention_pct=100.0),
                    CohortWeekData(week=1, retention_pct=85.0),
                ]),
            ],
        )

        assert len(data.cohorts) == 1
        assert len(data.cohorts[0].weeks) == 2
        assert data.cohorts[0].weeks[0].week == 0
        assert data.cohorts[0].weeks[1].retention_pct == 85.0

    def test_cohort_retention_pct_values(self):
        """Test that retention_pct accepts the full 0-100 range."""
        data = CohortRetentionData(cohorts=[
            Cohort(cohort_week="2024-01", weeks=[
                CohortWeekData(week=0, retention_pct=0.0),
                CohortWeekData(week=1, retention_pct=50.5),
                CohortWeekData(week=2, retention_pct=100.0),
            ]),
        ])
        weeks = data.cohorts[0].weeks
        assert weeks[0].retention_pct == 0.0
        assert weeks[1].retention_pct == 50.5
        assert weeks[2].retention_pct == 100.0

    def test_cohort_retention_serialization(self):
        """Test that CohortRetentionData serializes to the expected dict shape."""
        data = CohortRetentionData(cohorts=[
            Cohort(cohort_week="2024-01", weeks=[
                CohortWeekData(week=0, retention_pct=100.0),
            ]),
        ])

        serialized = data.model_dump()
        assert serialized == {
            "cohorts": [
                {"cohort_week": "2024-01", "weeks": [{"week": 0, "retention_pct": 100.0}]}
            ]
        }

    def test_cohort_retention_variable_week_counts(self):
        """Test that cohorts can have differing numbers of weeks."""
        data = CohortRetentionData(cohorts=[
            Cohort(cohort_week="2024-01", weeks=[
                CohortWeekData(week=0, retention_pct=100.0),
                CohortWeekData(week=1, retention_pct=85.0),
            ]),
            Cohort(cohort_week="2024-02", weeks=[
                CohortWeekData(week=0, retention_pct=98.0),
                CohortWeekData(week=1, retention_pct=80.0),
                CohortWeekData(week=2, retention_pct=65.0),
            ]),
        ])

        assert len(data.cohorts[0].weeks) == 2
        assert len(data.cohorts[1].weeks) == 3


class TestSchemaFieldValidation:
    """Test field validation across schemas."""

    def test_risk_response_missing_required_field(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            RiskResponse(
                user_id="user-123",
                risk_score=75,
                # Missing risk_tier and other required fields
            )

    def test_risk_summary_missing_required_field(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            RiskSummary(
                user_id="user-123",
                risk_score=75,
                # Missing risk_tier
            )

    def test_cohort_missing_required_field(self):
        """Test that missing required fields on Cohort fail validation."""
        with pytest.raises(ValidationError):
            Cohort(cohort_week="2024-01")  # Missing weeks

        with pytest.raises(ValidationError):
            CohortWeekData(week=0)  # Missing retention_pct
