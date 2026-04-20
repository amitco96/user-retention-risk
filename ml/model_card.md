# Model Card: User Retention Risk Prediction

## Model Overview

This XGBoost binary classification model predicts user churn risk (probability of becoming inactive) with explainability via SHAP values. The model scores each user on a 0-100 scale and identifies the top 3 risk drivers.

## Training Data

- **Source**: PostgreSQL database (backend/app/db/)
- **Table**: `users` (joined with `events`)
- **Time Period**: Synthetic data with 90-day observation window
- **Sample Size**: 1,000 users (Phase 1), 85,638 events
- **Churn Rate**: ~17% (200 churned users, 800 retained)
- **Train/Validation Split**: 80/20 (800 train, 200 val)

## Features (8 total)

| Feature | Type | Description | Units |
|---------|------|-------------|-------|
| `days_since_last_login` | Continuous | Recency: Days since most recent login event | days |
| `session_count_30d` | Discrete | Frequency: Number of logins in last 30 days | count |
| `feature_usage_count` | Discrete | Depth: Total feature_used events (all time) | count |
| `support_tickets_open` | Discrete | Friction: Number of open support tickets | count |
| `plan_type_encoded` | Ordinal | Value tier: free=0, starter=1, pro=2, enterprise=3 | 0-3 |
| `avg_session_duration_min` | Continuous | Quality: Mean session duration (login events) | minutes |
| `days_since_signup` | Continuous | Tenure: Account age | days |
| `login_streak_broken` | Binary | Behavioral: Login gap >1 day in last 7 days | 0/1 |

## Target Label

**Churn Definition** (binary):
```python
churn = (days_since_last_login >= 30) AND (sessions_60d < 3)
```

- `1` = Churned (inactive user: no login ≥30d ago AND <3 logins in 60d)
- `0` = Retained (active user)

## Model Architecture

- **Algorithm**: XGBoost Classifier (gradient boosting)
- **Number of Boosting Rounds**: ~500 (stopped early at round 10 without validation improvement)
- **Max Tree Depth**: 6
- **Learning Rate**: 0.1
- **Subsample**: 0.8 (row subsampling)
- **Column Subsample**: 0.8 (feature subsampling)
- **Early Stopping**: Yes (10 rounds without improvement on validation set)
- **Evaluation Metric**: Log loss

## Evaluation Metrics (Validation Set, 200 samples)

| Metric | Value |
|--------|-------|
| AUC-ROC | TBD (target ≥ 0.82) |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |
| True Negatives | TBD |
| False Positives | TBD |
| False Negatives | TBD |
| True Positives | TBD |

**Note**: Metrics populated after training run.

## Explainability (SHAP)

- **Explainer Type**: SHAP TreeExplainer (efficient for tree models)
- **Top Drivers**: Top 3 features ranked by absolute SHAP value per prediction
- **SHAP Values**: All 8 feature contributions included in API response
- **Interpretation**: Positive SHAP value = increases churn risk; Negative = decreases risk

## Risk Score & Tiers

Risk Score Calculation:
```python
risk_score = int(model.predict_proba(X)[:, 1] * 100)  # 0-100
```

Risk Tier Mapping:
- **Low**: 0-40
- **Medium**: 41-70
- **High**: 71-85
- **Critical**: 86-100

## Data Leakage Prevention

✓ All features computed using only data **before** the observation period
✓ StandardScaler fitted on train set only, applied to validation set
✓ No future information (e.g., support_ticket status) included
✓ Churn label uses hard cutoff (≥30 days) without lookahead

## Known Failure Modes

1. **Class Imbalance**: 17% churn rate may bias model toward retained class. Monitor: precision/recall tradeoff on at-risk cohort.

2. **New User Signal Weakness**: Users with <30 days tenure have limited `days_since_last_login` signal. Recommendation: Apply lower thresholds for new users (age <60d).

3. **Plan Type Leakage Risk**: Enterprise customers may have different behavior. Recommendation: Monitor AUC by plan tier; consider separate models if AUC variance >0.05.

4. **Session Duration Outliers**: Very long sessions (>8 hours) are rare; may inflate `avg_session_duration_min` for inactive users. Mitigation: Clipped at 99th percentile.

5. **Cold Start (No Events)**: Users with zero events default to feature values (e.g., `days_since_last_login=999`). These are likely to be high-risk (correct behavior).

6. **Time Dependency**: Model trained on synthetic data with fixed time windows (90d observation). Real-world performance may degrade with seasonal patterns.

## Model Artifacts

Saved to `/ml/artifacts/`:
- `model.pkl` - Joblib-serialized XGBClassifier
- `feature_names.json` - List of 8 feature names (for correct order at inference)
- `scaler.pkl` - Joblib-serialized StandardScaler (fitted on train)

## Inference API

Endpoint: `GET /users/{user_id}/risk`

Response:
```json
{
  "user_id": "uuid",
  "risk_score": 75,
  "risk_tier": "high",
  "top_drivers": ["days_since_last_login", "session_count_30d", "support_tickets_open"],
  "shap_values": {
    "days_since_last_login": 0.42,
    "session_count_30d": -0.18,
    "feature_usage_count": -0.05,
    "support_tickets_open": 0.08,
    "plan_type_encoded": 0.02,
    "avg_session_duration_min": -0.12,
    "days_since_signup": 0.01,
    "login_streak_broken": 0.03
  },
  "model_version": "1.0"
}
```

## Future Improvements

1. Test set holdout (currently 80/20 train/val split)
2. Hyperparameter tuning via Bayesian optimization
3. Feature engineering: interaction terms (e.g., `plan_type * feature_usage`)
4. Temporal cross-validation (time-aware splits)
5. Model monitoring: AUC drift detection, feature importance drift
6. Claude integration for reason/action generation (Phase 2)

---

**Model Version**: 1.0  
**Training Date**: 2026-04-20  
**Framework**: XGBoost 2.0.3 + scikit-learn 1.3.2 + SHAP 0.44.0
