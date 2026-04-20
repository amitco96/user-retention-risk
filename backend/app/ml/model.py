"""
XGBoost model wrapper and SHAP explainability.

Loads the trained model and provides RiskPrediction dataclass with:
- risk_score (0-100)
- risk_tier (low/medium/high/critical)
- top_drivers (top 3 SHAP feature names)
- shap_values (all feature SHAP values)
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import joblib
import shap


# Load model and scaler at module import time
_model_path = Path(__file__).parent.parent.parent.parent / "ml" / "artifacts" / "model.pkl"
_scaler_path = Path(__file__).parent.parent.parent.parent / "ml" / "artifacts" / "scaler.pkl"
_feature_names_path = Path(__file__).parent.parent.parent.parent / "ml" / "artifacts" / "feature_names.json"

# Load artifacts
_model = None
_scaler = None
_feature_names = None

try:
    _model = joblib.load(_model_path)
    _scaler = joblib.load(_scaler_path)
    with open(_feature_names_path, "r") as f:
        _feature_names = json.load(f)
except Exception as e:
    # Artifacts not yet available (will be loaded after training)
    print(f"Warning: Could not load model artifacts at import time: {e}")


@dataclass
class RiskPrediction:
    """Risk prediction result with SHAP explanations."""

    user_id: str
    risk_score: int  # 0-100
    risk_tier: str  # low, medium, high, critical
    top_drivers: List[str]  # top 3 feature names by SHAP
    shap_values: Dict[str, float]  # all feature SHAP values
    model_version: str = "1.0"


def _get_risk_tier(risk_score: int) -> str:
    """
    Map risk score to risk tier.

    Args:
        risk_score: Integer risk score (0-100)

    Returns:
        Risk tier: "low", "medium", "high", or "critical"
    """
    if risk_score <= 40:
        return "low"
    elif risk_score <= 70:
        return "medium"
    elif risk_score <= 85:
        return "high"
    else:
        return "critical"


def predict(user_id: str, features: Dict[str, float]) -> RiskPrediction:
    """
    Predict churn risk for a single user.

    Args:
        user_id: User identifier
        features: Dict of feature_name -> value with all 8 required features

    Returns:
        RiskPrediction dataclass with risk_score, risk_tier, top_drivers, shap_values

    Raises:
        ValueError: If model artifacts are not loaded or features are missing
    """
    if _model is None or _scaler is None or _feature_names is None:
        raise ValueError("Model artifacts not loaded. Run training first.")

    # Build feature vector in correct order
    X = np.array(
        [[features.get(fname, 0.0) for fname in _feature_names]], dtype=np.float32
    )

    # Scale features
    X_scaled = _scaler.transform(X)

    # Get probability prediction
    risk_proba = _model.predict_proba(X_scaled)[0, 1]  # Probability of churn
    risk_score = int(risk_proba * 100)

    # Get SHAP values
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_scaled)

    # Handle binary classification output
    # shap_values is a list [shap_values_class0, shap_values_class1]
    # We care about class 1 (churn)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # First sample, churn class
    else:
        shap_vals = shap_values[0]  # Single output

    # Map SHAP values to feature names
    shap_dict = {fname: float(shap_vals[i]) for i, fname in enumerate(_feature_names)}

    # Get top 3 drivers by absolute SHAP value
    sorted_drivers = sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_drivers = [fname for fname, _ in sorted_drivers[:3]]

    # Determine risk tier
    risk_tier = _get_risk_tier(risk_score)

    return RiskPrediction(
        user_id=user_id,
        risk_score=risk_score,
        risk_tier=risk_tier,
        top_drivers=top_drivers,
        shap_values=shap_dict,
        model_version="1.0",
    )
