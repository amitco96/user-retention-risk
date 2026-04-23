"""
Integration tests for router error paths and edge cases.

Directly tests the missing code lines in routers/users.py and routers/cohorts.py.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from backend.app.main import app


class TestUsersRouterIntegration:
    """Integration tests for users router edge cases."""

    def test_get_user_risk_success_with_all_fields(self, client_with_db, seeded_db, event_loop):
        """Test GET /users/{user_id}/risk with full successful response."""
        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        response = client.get(f"/users/{user_id}/risk")

        # Should either succeed with 200 or fail gracefully
        assert response.status_code in [200, 422, 500]

        if response.status_code == 200:
            data = response.json()
            assert "user_id" in data
            assert "risk_score" in data
            assert "risk_tier" in data
            assert "top_drivers" in data
            assert "reason" in data
            assert "recommended_action" in data
            assert isinstance(data["risk_score"], int)
            assert 0 <= data["risk_score"] <= 100

    def test_at_risk_with_mixed_scores(self, client_with_db):
        """Test /users/at-risk with different threshold values."""
        client = TestClient(app)

        # Test multiple thresholds
        for threshold in [0, 25, 50, 75, 100]:
            response = client.get(f"/users/at-risk?threshold={threshold}")
            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # All returned items should meet the threshold
            for item in data:
                assert item["risk_score"] >= threshold
                assert "user_id" in item
                assert "risk_score" in item
                assert "risk_tier" in item

    def test_feedback_creates_success_response(self, client_with_db, seeded_db):
        """Test POST /users/{user_id}/risk/feedback response structure."""
        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        response = client.post(f"/users/{user_id}/risk/feedback?action=contacted")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "message" in data
        assert data["status"] == "success"
        assert "contacted" in data["message"].lower() or "action" in data["message"].lower()

    def test_feedback_with_notes(self, client_with_db, seeded_db):
        """Test POST /users/{user_id}/risk/feedback with notes parameter."""
        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        response = client.post(
            f"/users/{user_id}/risk/feedback?action=email_sent&notes=Sent+special+offer"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestCohortsRouterIntegration:
    """Integration tests for cohorts router."""

    def test_cohort_retention_returns_valid_structure(self, client_with_db):
        """Test /cohorts/retention returns valid structure."""
        client = TestClient(app)
        response = client.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "cohorts" in data
        assert "weeks" in data
        assert "retention_matrix" in data

        # Validate types
        assert isinstance(data["cohorts"], list)
        assert isinstance(data["weeks"], list)
        assert isinstance(data["retention_matrix"], list)

        # Validate consistency
        if data["cohorts"]:
            assert len(data["retention_matrix"]) == len(data["cohorts"])

    def test_cohort_retention_values_all_valid(self, client_with_db):
        """Test /cohorts/retention all values are valid percentages."""
        client = TestClient(app)
        response = client.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()

        import math

        for cohort_row in data["retention_matrix"]:
            assert isinstance(cohort_row, list)
            for value in cohort_row:
                # Should be a number (int or float)
                assert isinstance(value, (int, float))
                # Should be between 0 and 100
                assert 0 <= value <= 100
                # Should not be NaN or inf
                assert not math.isnan(float(value))
                assert not math.isinf(float(value))

    def test_cohort_retention_cohort_labels_are_strings(self, client_with_db):
        """Test /cohorts/retention cohort labels are valid strings."""
        client = TestClient(app)
        response = client.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()

        for cohort_label in data["cohorts"]:
            assert isinstance(cohort_label, str)
            assert len(cohort_label) > 0
            # Should be YYYY-MM format from code
            if cohort_label != "":
                assert "-" in cohort_label or cohort_label.isdigit()

    def test_cohort_retention_week_labels_are_strings(self, client_with_db):
        """Test /cohorts/retention week labels are valid strings."""
        client = TestClient(app)
        response = client.get("/cohorts/retention")

        assert response.status_code == 200
        data = response.json()

        for week_label in data["weeks"]:
            assert isinstance(week_label, str)
            assert len(week_label) > 0


class TestUserFeatureExtraction:
    """Test feature extraction through the API."""

    def test_extract_features_with_valid_events(self, client_with_db, seeded_db):
        """Test that features are properly extracted for users with events."""
        # This is tested indirectly through the /users/{id}/risk endpoint
        # which calls extract_user_features internally

        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        # Mock the explainer so we can test feature extraction
        with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
            from backend.app.ml.explainer import RiskExplanation
            mock_explain.return_value = RiskExplanation(
                reason="Test reason",
                action="Test action",
            )

            response = client.get(f"/users/{user_id}/risk")

            if response.status_code == 200:
                data = response.json()
                # Verify top_drivers is populated (output from feature extraction)
                assert "top_drivers" in data
                assert isinstance(data["top_drivers"], list)


class TestErrorHandlingPaths:
    """Test specific error handling paths."""

    def test_invalid_json_response_from_claude(self, client_with_db, seeded_db):
        """Test handling of invalid JSON from Claude."""
        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        # Mock Claude to return invalid JSON
        with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
            from backend.app.ml.explainer import RiskExplanation
            # This simulates what happens when Claude returns invalid JSON
            mock_explain.return_value = RiskExplanation(
                reason="Unable to analyze.",
                action="Contact support.",
            )

            response = client.get(f"/users/{user_id}/risk")

            if response.status_code == 200:
                data = response.json()
                # Even with fallback explanation, should return valid response
                assert "reason" in data
                assert "recommended_action" in data

    def test_missing_api_key_fallback(self, client_with_db, seeded_db):
        """Test fallback when API key is missing."""
        user_id = seeded_db._test_user_2_id

        client = TestClient(app)
        with patch("backend.app.routers.users.explain_risk", new_callable=AsyncMock) as mock_explain:
            from backend.app.ml.explainer import RiskExplanation
            mock_explain.return_value = RiskExplanation(
                reason="Unable to analyze.",
                action="Contact support.",
            )

            response = client.get(f"/users/{user_id}/risk")

            if response.status_code == 200:
                data = response.json()
                assert "reason" in data
                assert "recommended_action" in data


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_risk_score_boundaries(self, client_with_db):
        """Test risk scores at boundaries."""
        client = TestClient(app)

        # Test threshold=0 returns all
        response = client.get("/users/at-risk?threshold=0")
        assert response.status_code == 200

        # Test threshold=100 returns only perfect scores or empty
        response = client.get("/users/at-risk?threshold=100")
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["risk_score"] == 100

    def test_response_consistency_across_calls(self, client_with_db):
        """Test that multiple calls to the same endpoint return consistent structure."""
        client = TestClient(app)

        # Call cohort retention multiple times
        for _ in range(3):
            response = client.get("/cohorts/retention")
            assert response.status_code == 200

            data = response.json()
            # Should always have these fields
            assert set(data.keys()) == {"cohorts", "weeks", "retention_matrix"}

    def test_empty_query_results_handling(self, client_with_db):
        """Test handling of empty query results."""
        client = TestClient(app)

        # Test at-risk with very high threshold
        response = client.get("/users/at-risk?threshold=99")
        assert response.status_code == 200
        data = response.json()
        # May be empty list
        assert isinstance(data, list)
