"""
XGBoost model training and evaluation script.

Trains churn prediction model on Sparkify dataset with early stopping,
evaluates on validation set, and saves artifacts (model.pkl, feature_names.json, scaler.pkl).
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


def load_recommended_features():
    """Load recommended features from Claude correlation report."""
    correlation_report_path = Path(__file__).parent / "artifacts" / "correlation_report.json"
    if not correlation_report_path.exists():
        print("WARNING: correlation_report.json not found, using all features")
        return ALL_FEATURE_NAMES

    with open(correlation_report_path, "r") as f:
        report = json.load(f)

    recommended = report.get("claude_analysis", {}).get("recommended_features", ALL_FEATURE_NAMES)
    return recommended


def train_model():
    """
    Train XGBoost model on Sparkify dataset with AI-recommended features.

    Pipeline:
    1. Load Sparkify dataset from ml/data/sparkify_mini.json
    2. Extract 8 Sparkify features per user
    3. Load recommended features from Claude correlation report
    4. Filter to only recommended features
    5. Train/validation split (80/20)
    6. Scale features (fit on train only)
    7. Train XGBoost
    8. Evaluate on validation set
    9. Save artifacts: model.pkl, feature_names.json, scaler.pkl
    """
    print("=" * 70)
    print("USER RETENTION RISK MODEL - SPARKIFY TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Load Sparkify dataset
    print("\n[1/7] Loading Sparkify dataset...")
    try:
        df = get_sparkify_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo train the model, download the Sparkify dataset:")
        print("  1. Download sparkify_mini.json from Kaggle or Udacity")
        print("  2. Create directory: mkdir -p ml/data")
        print("  3. Place file at: ml/data/sparkify_mini.json")
        sys.exit(1)

    print(f"     Loaded {len(df):,} events from {df['userId'].nunique():,} users")
    print(f"     Date range: {df['ts'].min()} to {df['ts'].max()}")

    # Step 2: Extract all features
    print("\n[2/7] Extracting features from Sparkify events...")
    X, y, all_feature_names = get_features(df)
    print(f"     Feature matrix shape: {X.shape}")
    print(f"     Churn rate: {y.mean():.4f}")
    print(f"     All features: {all_feature_names}")

    # Step 3: Load recommended features from Claude correlation report
    print("\n[3/7] Loading Claude-recommended features...")
    recommended_features = load_recommended_features()
    print(f"     Using {len(recommended_features)} features: {recommended_features}")

    # Filter features to recommended set
    feature_indices = [all_feature_names.index(f) for f in recommended_features if f in all_feature_names]
    X = X[:, feature_indices]
    feature_names = [all_feature_names[i] for i in feature_indices]
    print(f"     Filtered feature matrix shape: {X.shape}")

    # Step 4: Train/validation split (80/20)
    print("\n[4/7] Splitting data into train (80%) and validation (20%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"     Train set: {X_train.shape[0]} samples, churn rate: {y_train.mean():.4f}")
    print(f"     Validation set: {X_val.shape[0]} samples, churn rate: {y_val.mean():.4f}")

    # Step 5: Fit StandardScaler on train only
    print("\n[5/7] Scaling features (fit on train, transform train + val)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"     Scaler fitted on train set")

    # Step 6: Train XGBoost
    print("\n[6/7] Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )

    xgb_model.fit(
        X_train_scaled,
        y_train,
        verbose=False,
    )
    print(f"     Model trained with {xgb_model.n_estimators} boosting rounds")

    # Step 7: Evaluate on validation set
    print("\n[7/7] Evaluating model on validation set...")
    y_val_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = xgb_model.predict(X_val_scaled)

    auc_roc = roc_auc_score(y_val, y_val_pred_proba)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_val_pred)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS (Validation Set)")
    print("=" * 70)
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"     True Negatives:  {cm[0, 0]}")
    print(f"     False Positives: {cm[0, 1]}")
    print(f"     False Negatives: {cm[1, 0]}")
    print(f"     True Positives:  {cm[1, 1]}")
    print("=" * 70)

    # Save artifacts
    print("\n[SAVE] Saving model artifacts...")
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save model
    model_path = artifacts_dir / "model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"     Model saved to: {model_path}")

    # Save feature names
    feature_names_path = artifacts_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)
    print(f"     Feature names saved to: {feature_names_path}")

    # Save scaler
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
    }


if __name__ == "__main__":
    metrics = train_model()
