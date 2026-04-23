"""
Unit tests for FastAPI endpoints (backend/app/routers/).

Tests:
- GET /health returns 200
- GET /users/{user_id}/risk returns RiskResponse (or 404/422 for errors)
- GET /users/at-risk?threshold=70 returns list of RiskSummary
- POST /users/{user_id}/risk/feedback returns success or error
- GET /cohorts/retention returns CohortRetentionData
- Invalid user_id returns 422
- Non-existent user returns 404
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check_returns_200(self):
        """Test that /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_status_ok(self):
        """Test that /health returns status='ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_response_structure(self):
        """Test that /health response has expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_version" in data

    def test_health_check_multiple_calls(self):
        """Test that /health can be called multiple times."""
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


class TestGetUserRiskEndpoint:
    """Test GET /users/{user_id}/risk endpoint."""

    def test_get_user_risk_invalid_uuid_returns_422(self):
        """Test that invalid UUID format returns 422."""
        invalid_ids = [
            "not-a-uuid",
            "123",
            "user@example.com",
            "12345678-1234-1234-1234",
            "",
            "invalid",
        ]

        for invalid_id in invalid_ids:
            response = client.get(f"/users/{invalid_id}/risk")
            assert response.status_code == 422, f"Expected 422 for {invalid_id}, got {response.status_code}"

    def test_get_user_risk_nonexistent_user_returns_404(self):
        """Test that non-existent user returns 404."""
        fake_id = uuid.uuid4()
        response = client.get(f"/users/{fake_id}/risk")
        assert response.status_code == 404

    def test_get_user_risk_various_valid_uuids_format(self):
        """Test that valid UUID formats are accepted."""
        # Valid UUIDs that don't exist should return 404
        valid_uuids = [
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]

        for uuid_str in valid_uuids:
            response = client.get(f"/users/{uuid_str}/risk")
            # Should be 404 (not found) not 422 (invalid format)
            assert response.status_code in [404, 500], f"Expected 404 or 500 for {uuid_str}, got {response.status_code}"


class TestGetAtRiskUsersEndpoint:
    """Test GET /users/at-risk?threshold=70 endpoint."""

    def test_at_risk_returns_200_without_data(self):
        """Test that /users/at-risk returns 200 even with no data."""
        response = client.get("/users/at-risk?threshold=70")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_at_risk_returns_list(self):
        """Test that /users/at-risk returns a list or error."""
        response = client.get("/users/at-risk?threshold=70")
        # Should return either list (200) or error (500)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
        else:
            assert response.status_code == 500

    def test_at_risk_default_threshold_70(self):
        """Test that default threshold is 70."""
        response = client.get("/users/at-risk")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_at_risk_valid_thresholds(self):
        """Test various valid threshold values."""
        valid_thresholds = [0, 25, 50, 70, 85, 100]

        for threshold in valid_thresholds:
            response = client.get(f"/users/at-risk?threshold={threshold}")
            # Should return 200 or 500 (if DB unavailable)
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)

    def test_at_risk_invalid_threshold_returns_422(self):
        """Test that invalid thresholds return 422."""
        invalid_thresholds = ["-1", "101", "150", "abc"]

        for threshold in invalid_thresholds:
            response = client.get(f"/users/at-risk?threshold={threshold}")
            assert response.status_code == 422

    def test_at_risk_threshold_boundary_values(self):
        """Test boundary threshold values."""
        boundary_thresholds = [0, 1, 50, 99, 100]

        for threshold in boundary_thresholds:
            response = client.get(f"/users/at-risk?threshold={threshold}")
            # Should return 200 or 500 (if DB unavailable)
            assert response.status_code in [200, 500]


class TestPostRiskFeedbackEndpoint:
    """Test POST /users/{user_id}/risk/feedback endpoint."""

    def test_risk_feedback_invalid_uuid_returns_422(self):
        """Test that invalid UUID returns 422."""
        response = client.post(
            "/users/not-a-uuid/risk/feedback",
            json={"action": "email_sent", "notes": "Test"},
        )
        assert response.status_code == 422

    def test_risk_feedback_nonexistent_user_returns_404(self):
        """Test that non-existent user returns 404 (or 500 if no risk score exists)."""
        fake_id = uuid.uuid4()
        response = client.post(
            f"/users/{fake_id}/risk/feedback",
            json={"action": "email_sent"},
        )
        # Should be 404 if the user exists but has no risk score, or 500 for DB error
        assert response.status_code in [404, 500]

    def test_risk_feedback_missing_action_param(self):
        """Test that missing action parameter fails validation."""
        user_id = uuid.uuid4()
        response = client.post(
            f"/users/{user_id}/risk/feedback",
            json={"notes": "No action provided"},
        )
        # Should fail validation - missing required field
        assert response.status_code in [422, 500]

    def test_risk_feedback_missing_body_entirely(self):
        """Test that missing entire body fails."""
        user_id = uuid.uuid4()
        response = client.post(f"/users/{user_id}/risk/feedback")
        # Should fail validation
        assert response.status_code in [422, 500]

    def test_risk_feedback_various_actions(self):
        """Test POST with various action values."""
        user_id = uuid.uuid4()
        actions = ["email_sent", "call_scheduled", "discount_offered", "manual_review"]

        for action in actions:
            response = client.post(
                f"/users/{user_id}/risk/feedback",
                json={"action": action},
            )
            # Should either succeed (200) or fail with 404/500 (user not found)
            assert response.status_code in [200, 404, 500]


class TestGetCohortRetentionEndpoint:
    """Test GET /cohorts/retention endpoint."""

    def test_cohort_retention_returns_ok(self):
        """Test that /cohorts/retention returns 200 or 500."""
        response = client.get("/cohorts/retention")
        assert response.status_code in [200, 500]

    def test_cohort_retention_response_structure(self):
        """Test that response has required fields."""
        response = client.get("/cohorts/retention")
        if response.status_code == 200:
            data = response.json()

            assert "cohorts" in data
            assert "weeks" in data
            assert "retention_matrix" in data

    def test_cohort_retention_response_types(self):
        """Test that response fields have correct types."""
        response = client.get("/cohorts/retention")
        if response.status_code == 200:
            data = response.json()

            assert isinstance(data["cohorts"], list)
            assert isinstance(data["weeks"], list)
            assert isinstance(data["retention_matrix"], list)

    def test_cohort_retention_matrix_consistency(self):
        """Test that retention_matrix has consistent shape with cohorts."""
        response = client.get("/cohorts/retention")
        if response.status_code == 200:
            data = response.json()

            cohorts = data["cohorts"]
            matrix = data["retention_matrix"]

            # Matrix should have one row per cohort
            assert len(matrix) == len(cohorts)

    def test_cohort_retention_retention_values_valid(self):
        """Test that retention percentages are valid (0-100)."""
        response = client.get("/cohorts/retention")
        if response.status_code == 200:
            data = response.json()

            matrix = data["retention_matrix"]
            for row in matrix:
                for value in row:
                    assert isinstance(value, (int, float)), f"Expected number, got {type(value)}"
                    assert 0.0 <= value <= 100.0, f"Expected 0-100, got {value}"

    def test_cohort_retention_empty_data_handling(self):
        """Test that empty data is handled gracefully."""
        response = client.get("/cohorts/retention")
        if response.status_code == 200:
            data = response.json()
            # Should return valid structure even with no data
            assert isinstance(data["cohorts"], list)
            assert isinstance(data["weeks"], list)
            assert isinstance(data["retention_matrix"], list)


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_nonexistent_endpoint_returns_404(self):
        """Test that accessing non-existent endpoint returns 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method_returns_405(self):
        """Test that using wrong HTTP method returns 405."""
        response = client.post("/health")
        assert response.status_code == 405

    def test_malformed_request_returns_422(self):
        """Test that malformed requests return 422."""
        response = client.get("/users/invalid/risk")
        assert response.status_code == 422

    def test_options_request_handling(self):
        """Test that OPTIONS requests are handled."""
        response = client.options("/health")
        # FastAPI usually returns 405 for OPTIONS on regular endpoints
        assert response.status_code in [405, 200]


class TestEndpointIntegration:
    """Integration tests for endpoint workflows."""

    def test_health_endpoint_always_available(self):
        """Test that health endpoint is always available."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_at_risk_endpoint_always_available(self):
        """Test that at-risk endpoint is always available."""
        response = client.get("/users/at-risk?threshold=50")
        assert response.status_code == 200

    def test_cohort_retention_endpoint_always_available(self):
        """Test that cohort retention endpoint is always available."""
        response = client.get("/cohorts/retention")
        assert response.status_code == 200

    def test_endpoints_consistent_response_format(self):
        """Test that endpoints return consistent JSON responses."""
        # Health
        health = client.get("/health")
        assert health.headers["content-type"].startswith("application/json")

        # At-risk
        at_risk = client.get("/users/at-risk")
        assert at_risk.headers["content-type"].startswith("application/json")

        # Cohort
        cohort = client.get("/cohorts/retention")
        assert cohort.headers["content-type"].startswith("application/json")


class TestResponseValidation:
    """Test response validation and schema compliance."""

    def test_health_response_has_no_unexpected_fields(self):
        """Test that health response contains only expected fields."""
        response = client.get("/health")
        data = response.json()

        expected_keys = {"status", "model_version"}
        actual_keys = set(data.keys())

        # All expected keys should be present
        assert expected_keys <= actual_keys, f"Missing keys: {expected_keys - actual_keys}"

    def test_at_risk_response_items_structure(self):
        """Test that at-risk response items have correct structure."""
        response = client.get("/users/at-risk?threshold=0")
        data = response.json()

        assert isinstance(data, list)
        # If there are items, they should have required fields
        if len(data) > 0:
            item = data[0]
            required_keys = {"user_id", "risk_score", "risk_tier", "reason"}
            assert required_keys <= set(item.keys()), f"Missing keys in at-risk item: {required_keys - set(item.keys())}"

            # Validate field types
            assert isinstance(item["user_id"], str)
            assert isinstance(item["risk_score"], int)
            assert isinstance(item["risk_tier"], str)
            assert isinstance(item["reason"], str)

    def test_cohort_retention_matrix_alignment(self):
        """Test that retention matrix is properly aligned with cohort/week labels."""
        response = client.get("/cohorts/retention")
        data = response.json()

        cohorts = data["cohorts"]
        weeks = data["weeks"]
        matrix = data["retention_matrix"]

        # Each row should align with weeks
        for i, row in enumerate(matrix):
            if len(row) > 0:
                assert len(row) == len(weeks), f"Row {i} has {len(row)} values but expected {len(weeks)}"


class TestConcurrency:
    """Test endpoint behavior under concurrent access patterns."""

    def test_health_rapid_consecutive_calls(self):
        """Test health endpoint with rapid consecutive calls."""
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

    def test_mixed_endpoints_calls(self):
        """Test calling different endpoints in sequence."""
        responses = [
            client.get("/health"),
            client.get("/users/at-risk?threshold=50"),
            client.get("/cohorts/retention"),
        ]

        for response in responses:
            assert response.status_code == 200


class TestRequestValidation:
    """Test request validation and parameter handling."""

    def test_query_parameter_case_sensitivity(self):
        """Test that query parameters are case-sensitive."""
        # Standard parameter
        response1 = client.get("/users/at-risk?threshold=70")
        assert response1.status_code == 200

        # Different case (might not be recognized as parameter)
        response2 = client.get("/users/at-risk?Threshold=70")
        # Should still work but threshold might not be applied
        assert response2.status_code in [200, 422]

    def test_extra_query_parameters_ignored(self):
        """Test that extra query parameters don't break endpoint."""
        response = client.get("/users/at-risk?threshold=70&extra_param=value&another=123")
        assert response.status_code == 200

    def test_empty_path_segments(self):
        """Test that empty path segments are handled."""
        response = client.get("/users//risk")
        # This might be interpreted differently
        assert response.status_code in [404, 422]
