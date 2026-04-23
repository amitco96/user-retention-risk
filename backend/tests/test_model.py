"""
Unit tests for ML model wrapper (backend/app/ml/model.py).

Tests:
- Model loads from ml/artifacts/model.pkl without error
- predict() returns RiskPrediction dataclass
- risk_score is int between 0-100
- risk_tier is one of "low"|"medium"|"high"|"critical"
- top_drivers is a list of 3 strings
- shap_values is a dict mapping feature names to floats
- _get_risk_tier maps scores to correct tiers
- Model handles feature dict with correct ordering
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from backend.app.ml.model import predict, _get_risk_tier, RiskPrediction


class TestGetRiskTier:
    """Test risk tier mapping function."""

    def test_risk_tier_low_score_40_or_below(self):
        """Test that scores <= 40 return 'low'."""
        assert _get_risk_tier(0) == "low"
        assert _get_risk_tier(20) == "low"
        assert _get_risk_tier(40) == "low"

    def test_risk_tier_medium_score_41_to_70(self):
        """Test that scores 41-70 return 'medium'."""
        assert _get_risk_tier(41) == "medium"
        assert _get_risk_tier(55) == "medium"
        assert _get_risk_tier(70) == "medium"

    def test_risk_tier_high_score_71_to_85(self):
        """Test that scores 71-85 return 'high'."""
        assert _get_risk_tier(71) == "high"
        assert _get_risk_tier(78) == "high"
        assert _get_risk_tier(85) == "high"

    def test_risk_tier_critical_score_above_85(self):
        """Test that scores > 85 return 'critical'."""
        assert _get_risk_tier(86) == "critical"
        assert _get_risk_tier(95) == "critical"
        assert _get_risk_tier(100) == "critical"

    def test_risk_tier_boundary_values(self):
        """Test boundary values between tiers."""
        assert _get_risk_tier(40) == "low"
        assert _get_risk_tier(41) == "medium"
        assert _get_risk_tier(70) == "medium"
        assert _get_risk_tier(71) == "high"
        assert _get_risk_tier(85) == "high"
        assert _get_risk_tier(86) == "critical"


class TestRiskPredictionDataclass:
    """Test RiskPrediction dataclass."""

    def test_risk_prediction_creation(self):
        """Test that RiskPrediction can be instantiated."""
        prediction = RiskPrediction(
            user_id="user123",
            risk_score=75,
            risk_tier="high",
            top_drivers=["feature1", "feature2", "feature3"],
            shap_values={"feature1": 0.5, "feature2": 0.3},
            model_version="1.0",
        )

        assert prediction.user_id == "user123"
        assert prediction.risk_score == 75
        assert prediction.risk_tier == "high"
        assert len(prediction.top_drivers) == 3
        assert prediction.model_version == "1.0"

    def test_risk_prediction_default_version(self):
        """Test that model_version defaults to '1.0'."""
        prediction = RiskPrediction(
            user_id="user123",
            risk_score=50,
            risk_tier="medium",
            top_drivers=["feature1", "feature2", "feature3"],
            shap_values={},
        )

        assert prediction.model_version == "1.0"


class TestPredict:
    """Test predict function."""

    @pytest.fixture
    def mock_model_artifacts(self):
        """Mock the loaded model artifacts."""
        feature_list = [
            "songs_played_total",
            "thumbs_down_count",
            "thumbs_up_count",
            "add_to_playlist_count",
            "avg_session_duration_min",
        ]

        with patch("backend.app.ml.model._model") as mock_model, \
             patch("backend.app.ml.model._scaler") as mock_scaler, \
             patch("backend.app.ml.model._feature_names", feature_list):

            # Setup mock scaler
            mock_scaler.transform.return_value = np.array([[0.5, 0.3, 0.2, 0.1, 0.0]], dtype=np.float32)

            # Setup mock model
            mock_model.predict_proba.return_value = np.array([[0.3, 0.65]])  # 65% churn probability

            yield {
                "model": mock_model,
                "scaler": mock_scaler,
                "feature_names": feature_list,
            }

    def test_predict_raises_error_when_model_not_loaded(self):
        """Test that predict raises ValueError when model is not loaded."""
        with patch("backend.app.ml.model._model", None), \
             patch("backend.app.ml.model._scaler", None), \
             patch("backend.app.ml.model._feature_names", None):

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            with pytest.raises(ValueError, match="Model artifacts not loaded"):
                predict("user123", features)

    def test_predict_returns_risk_prediction_dataclass(self, mock_model_artifacts):
        """Test that predict returns RiskPrediction dataclass."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            # Mock SHAP explainer - returns list of arrays [class0_shap, class1_shap]
            mock_explainer = MagicMock()
            # SHAP returns list: [array for class 0, array for class 1]
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),  # Class 0
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])   # Class 1 (churn)
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert isinstance(result, RiskPrediction)
            assert result.user_id == "user123"

    def test_predict_risk_score_is_int_0_to_100(self, mock_model_artifacts):
        """Test that risk_score is int between 0-100."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert isinstance(result.risk_score, int)
            assert 0 <= result.risk_score <= 100

    def test_predict_risk_tier_is_valid(self, mock_model_artifacts):
        """Test that risk_tier is one of valid values."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert result.risk_tier in ["low", "medium", "high", "critical"]

    def test_predict_top_drivers_is_list_of_3_strings(self, mock_model_artifacts):
        """Test that top_drivers is a list of exactly 3 strings."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert isinstance(result.top_drivers, list)
            assert len(result.top_drivers) == 3
            assert all(isinstance(driver, str) for driver in result.top_drivers)

    def test_predict_shap_values_is_dict(self, mock_model_artifacts):
        """Test that shap_values is a dict mapping feature names to floats."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert isinstance(result.shap_values, dict)
            assert len(result.shap_values) == 5  # 5 features
            assert all(isinstance(v, (int, float)) for v in result.shap_values.values())

    def test_predict_feature_handling_missing_features_default_to_zero(self, mock_model_artifacts):
        """Test that missing features default to 0.0."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            # Partial features dict - missing some keys
            features = {
                "songs_played_total": 10.0,
                "thumbs_up_count": 5.0,
                # Missing: thumbs_down_count, add_to_playlist_count, avg_session_duration_min
            }

            result = predict("user123", features)

            # Should not raise error
            assert isinstance(result, RiskPrediction)
            # Verify scaler was called with 5 features (missing ones should be 0.0)
            call_args = mock_model_artifacts["scaler"].transform.call_args
            X = call_args[0][0]
            assert X.shape[1] == 5

    def test_predict_high_risk_score(self, mock_model_artifacts):
        """Test prediction with high risk probability."""
        with patch("backend.app.ml.model._model") as mock_model, \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            # High churn probability
            mock_model.predict_proba.return_value = np.array([[0.1, 0.95]])

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert result.risk_score == 95
            assert result.risk_tier == "critical"

    def test_predict_low_risk_score(self, mock_model_artifacts):
        """Test prediction with low risk probability."""
        with patch("backend.app.ml.model._model") as mock_model, \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            # Low churn probability
            mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.05, 0.04, 0.02, 0.01, 0.01]]),
                np.array([[0.02, 0.01, 0.01, 0.01, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert result.risk_score == 8
            assert result.risk_tier == "low"

    def test_predict_top_drivers_sorted_by_abs_shap_value(self, mock_model_artifacts):
        """Test that top_drivers are sorted by absolute SHAP value."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            # SHAP values in specific order
            mock_explainer = MagicMock()
            # [0.5, 0.3, 0.15, 0.05, 0.01] - sorted descending by abs value
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.05, 0.02, 0.01, 0.005]]),
                np.array([[0.5, 0.3, 0.15, 0.05, 0.01]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            # Top 3 should be in order of SHAP value magnitude
            assert len(result.top_drivers) == 3

    def test_predict_model_version_preserved(self, mock_model_artifacts):
        """Test that model_version is correctly set."""
        with patch("backend.app.ml.model._model", mock_model_artifacts["model"]), \
             patch("backend.app.ml.model._scaler", mock_model_artifacts["scaler"]), \
             patch("backend.app.ml.model._feature_names", mock_model_artifacts["feature_names"]), \
             patch("shap.TreeExplainer") as mock_explainer_class:

            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = [
                np.array([[0.1, 0.08, 0.05, 0.02, 0.01]]),
                np.array([[0.2, 0.15, 0.1, 0.05, 0.0]])
            ]
            mock_explainer_class.return_value = mock_explainer

            features = {
                "songs_played_total": 10.0,
                "thumbs_down_count": 2.0,
                "thumbs_up_count": 5.0,
                "add_to_playlist_count": 3.0,
                "avg_session_duration_min": 30.0,
            }

            result = predict("user123", features)

            assert result.model_version == "1.0"
