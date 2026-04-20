"""
XGBoost model training and evaluation script.

Trains the churn prediction model with early stopping, evaluates on validation set,
and saves artifacts (model.pkl, feature_names.json).
"""

import asyncio
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

from ml.feature_engineering import get_features, FEATURE_NAMES


async def train_model():
    """
    Train XGBoost model with early stopping and save artifacts.
    """
    print("=" * 70)
    print("USER RETENTION RISK MODEL - TRAINING PIPELINE")
    print("=" * 70)

    # Step 1: Extract features
    print("\n[1/5] Extracting features from PostgreSQL...")
    X, y, feature_names = await get_features()
    print(f"     Feature matrix shape: {X.shape}")
    print(f"     Churn rate: {y.mean():.4f}")
    print(f"     Features: {feature_names}")

    # Step 2: Train/validation split (80/20)
    print("\n[2/5] Splitting data into train (80%) and validation (20%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"     Train set: {X_train.shape[0]} samples, churn rate: {y_train.mean():.4f}")
    print(f"     Validation set: {X_val.shape[0]} samples, churn rate: {y_val.mean():.4f}")

    # Step 3: Fit StandardScaler on train only
    print("\n[3/5] Scaling features (fit on train, transform train + val)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"     Scaler fitted on train set")

    # Step 4: Train XGBoost with early stopping
    print("\n[4/5] Training XGBoost model with early stopping...")
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

    # Train with early stopping on validation set
    xgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        early_stopping_rounds=10,
        verbose=False,
    )
    print(f"     Model trained with {xgb_model.n_estimators} boosting rounds")

    # Step 5: Evaluate on validation set
    print("\n[5/5] Evaluating model on validation set...")
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
    metrics = asyncio.run(train_model())
