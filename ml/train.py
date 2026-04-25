"""
XGBoost model training and evaluation script.

Trains churn prediction model on the medium Sparkify dataset, evaluates on a
stratified test split, and saves artifacts (model.pkl, scaler.pkl,
feature_names.json) under ml/artifacts/.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import xgboost as xgb
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_loader import get_sparkify_data
from ml.feature_engineering import get_features, FEATURE_NAMES as ALL_FEATURE_NAMES


def _compute_top_shap_features(model, X_scaled, feature_names, top_n=5):
    """Return top-N feature names ranked by mean |SHAP| on the supplied set."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            # Old SHAP API for binary classifiers returned [class0, class1]
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return [(feature_names[i], float(mean_abs[i])) for i in order[:top_n]]
    except Exception as e:
        print(f"     SHAP unavailable, falling back to gain importance: {e}")
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        return [(feature_names[i], float(importances[i])) for i in order[:top_n]]


def train_model():
    """
    Train XGBoost model on the medium Sparkify dataset using all 8 features.

    Pipeline:
    1. Load Sparkify dataset (medium-sparkify-event-data.json)
    2. Extract 8 Sparkify features per user
    3. Stratified 80/20 train/test split (random_state=42)
    4. StandardScaler fit on train, transform both
    5. Train XGBoost with scale_pos_weight (auto class imbalance correction)
    6. Evaluate on test set: AUC-ROC, precision, recall, F1, confusion matrix
    7. Save model.pkl, scaler.pkl, feature_names.json
    """
    print("=" * 70)
    print("USER RETENTION RISK MODEL - SPARKIFY (MEDIUM) TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load Sparkify dataset
    print("\n[1/6] Loading Sparkify dataset...")
    try:
        df = get_sparkify_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"     Loaded {len(df):,} events from {df['userId'].nunique():,} users")
    print(f"     Date range: {df['ts'].min()} to {df['ts'].max()}")

    # Step 2: Extract all 8 features
    print("\n[2/6] Extracting 8 features from Sparkify events...")
    X, y, feature_names = get_features(df)
    print(f"     Feature matrix shape: {X.shape}")
    print(f"     Churn rate: {y.mean():.4f}")
    print(f"     Features: {feature_names}")

    # Step 3: Stratified 80/20 train/test split
    print("\n[3/6] Stratified 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"     Train set: {X_train.shape[0]} samples, churn rate: {y_train.mean():.4f}")
    print(f"     Test set:  {X_test.shape[0]} samples, churn rate: {y_test.mean():.4f}")

    # Step 4: Fit StandardScaler on train only
    print("\n[4/6] Scaling features (fit on train, transform train + test)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train XGBoost
    print("\n[5/6] Training XGBoost model...")
    # scale_pos_weight = neg_count / pos_count to correct class imbalance
    pos_count = max(1, int((y_train == 1).sum()))
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"     scale_pos_weight={scale_pos_weight:.3f} (neg={neg_count}, pos={pos_count})")

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        eval_metric="auc",
        # XGBoost 2.x: early_stopping_rounds is set on the constructor
        early_stopping_rounds=20,
    )

    xgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    # `best_iteration` is set when early stopping triggers
    n_used = getattr(xgb_model, "best_iteration", None)
    if n_used is None:
        n_used = xgb_model.n_estimators
    print(f"     Trained with {n_used + 1 if isinstance(n_used, int) else n_used} boosting rounds (best)")

    # Step 6: Evaluate on test set
    print("\n[6/6] Evaluating model on test set...")
    y_test_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = xgb_model.predict(X_test_scaled)

    auc_roc = roc_auc_score(y_test, y_test_pred_proba)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS (Test Set)")
    print("=" * 70)
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"     True Negatives:  {cm[0, 0]}")
    print(f"     False Positives: {cm[0, 1]}")
    print(f"     False Negatives: {cm[1, 0]}")
    print(f"     True Positives:  {cm[1, 1]}")
    print("=" * 70)

    # Top SHAP features (computed on the test split)
    print("\nTop SHAP features (mean |SHAP| on test set):")
    top_shap = _compute_top_shap_features(xgb_model, X_test_scaled, feature_names, top_n=8)
    for name, importance in top_shap:
        print(f"     {name:30s}  {importance:.4f}")

    # Hard requirement
    if auc_roc < 0.82:
        print(f"\nWARNING: AUC-ROC {auc_roc:.4f} is below required threshold 0.82")

    # Save artifacts
    print("\n[SAVE] Saving model artifacts...")
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"     Model saved to: {model_path}")

    feature_names_path = artifacts_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"     Feature names saved to: {feature_names_path}")

    scaler_path = artifacts_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"     Scaler saved to: {scaler_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return {
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "top_shap_features": top_shap,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "churn_rate": float(y.mean()),
        "n_users": int(X.shape[0]),
    }


if __name__ == "__main__":
    metrics = train_model()
