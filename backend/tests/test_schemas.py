"""
Unit tests for Pydantic schemas (backend/app/schemas/risk.py).

Tests schema validation and serialization.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from backend.app.schemas.risk import RiskResponse, RiskSummary, CohortRetentionData


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
    """Test CohortRetentionData schema."""

    def test_cohort_retention_data_valid(self):
        """Test CohortRetentionData with valid data."""
        data = CohortRetentionData(
            cohorts=["2024-01", "2024-02", "2024-03"],
            weeks=["Week 0", "Week 1", "Week 2"],
            retention_matrix=[
                [100.0, 85.0, 70.0],
                [98.0, 80.0, 65.0],
                [95.0, 75.0, 60.0],
            ],
        )

        assert len(data.cohorts) == 3
        assert len(data.weeks) == 3
        assert len(data.retention_matrix) == 3

    def test_cohort_retention_empty_data(self):
        """Test CohortRetentionData with empty data."""
        data = CohortRetentionData(
            cohorts=[],
            weeks=[],
            retention_matrix=[],
        )

        assert data.cohorts == []
        assert data.weeks == []
        assert data.retention_matrix == []

    def test_cohort_retention_single_cohort(self):
        """Test CohortRetentionData with single cohort."""
        data = CohortRetentionData(
            cohorts=["2024-01"],
            weeks=["Week 0", "Week 1"],
            retention_matrix=[[100.0, 85.0]],
        )

        assert len(data.cohorts) == 1
        assert len(data.weeks) == 2
        assert len(data.retention_matrix) == 1
        assert len(data.retention_matrix[0]) == 2

    def test_cohort_retention_retention_values(self):
        """Test that retention values can be 0-100."""
        data = CohortRetentionData(
            cohorts=["2024-01"],
            weeks=["Week 0"],
            retention_matrix=[[0.0]],
        )
        assert data.retention_matrix[0][0] == 0.0

        data = CohortRetentionData(
            cohorts=["2024-01"],
            weeks=["Week 0"],
            retention_matrix=[[100.0]],
        )
        assert data.retention_matrix[0][0] == 100.0

        data = CohortRetentionData(
            cohorts=["2024-01"],
            weeks=["Week 0"],
            retention_matrix=[[50.5]],
        )
        assert data.retention_matrix[0][0] == 50.5

    def test_cohort_retention_serialization(self):
        """Test that CohortRetentionData can be serialized."""
        data = CohortRetentionData(
            cohorts=["2024-01"],
            weeks=["Week 0"],
            retention_matrix=[[100.0]],
        )

        serialized = data.model_dump()
        assert serialized["cohorts"] == ["2024-01"]
        assert serialized["weeks"] == ["Week 0"]
        assert serialized["retention_matrix"] == [[100.0]]

    def test_cohort_retention_matrix_shape_flexibility(self):
        """Test that retention_matrix can have variable shape."""
        # Different number of weeks for each cohort is allowed
        data = CohortRetentionData(
            cohorts=["2024-01", "2024-02"],
            weeks=["Week 0", "Week 1", "Week 2"],
            retention_matrix=[
                [100.0, 85.0],  # 2 values for first cohort
                [98.0, 80.0, 65.0],  # 3 values for second cohort
            ],
        )

        assert len(data.retention_matrix) == 2


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

    def test_cohort_retention_missing_required_field(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            CohortRetentionData(
                cohorts=["2024-01"],
                # Missing weeks and retention_matrix
            )
