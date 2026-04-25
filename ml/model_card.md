# Model Card: User Retention Risk Prediction

## Model Overview

XGBoost binary classifier predicting user churn risk (probability of cancelling)
on a 0-100 scale, with SHAP-based explanations. The top 3 SHAP-ranked features
are surfaced per prediction so a CSM can see *why* a user is flagged.

## Training Data

- **Source**: Sparkify medium event-data dump (`ml/data/medium-sparkify-event-data.json`, ~232 MB JSONL)
- **Time window**: 2018-10-01 → 2018-12-01 (raw event timestamps)
- **Total events**: 528,005
- **Distinct users**: 448 (after dropping rows with empty `userId`)
- **Churn rate**: 22.10% (99 churned / 349 retained)
- **Train/test split**: stratified 80/20, `random_state=42`
  - Train: 358 users (churn rate 22.07%)
  - Test:  90 users (churn rate 22.22%)

Churn label: a user is `churn=1` if their `userId` ever appears on the
`Cancellation Confirmation` page.

## Features (8 total)

| Feature | Type | Description | Units |
|---|---|---|---|
| `days_since_last_activity` | Continuous | Days from observation date to most recent event | days |
| `session_count_30d` | Discrete | Distinct `sessionId` count in last 30 days (gap-clustered at runtime when sessionId is unavailable, e.g. live Postgres events) | count |
| `songs_played_total` | Discrete | Total `NextSong` events all-time (live: `feature_used` events) | count |
| `thumbs_up_count` | Discrete | `Thumbs Up` events (live: `support_ticket` with sentiment=positive) | count |
| `thumbs_down_count` | Discrete | `Thumbs Down` events (live: `support_ticket` with sentiment=negative) | count |
| `add_to_playlist_count` | Discrete | `Add to Playlist` events (live: `feature_used` with feature_name=playlist) | count |
| `avg_session_duration_min` | Continuous | Mean intra-session span in minutes | minutes |
| `subscription_level` | Binary | `free`=0, `paid`/`starter`/`pro`/`enterprise`=1 | 0/1 |

### Imputation strategy

- Users with no events are skipped during training (label undefined).
- `days_since_last_activity` is clamped to 0 when `observation_date` precedes
  the most recent event (clock skew tolerance).
- `subscription_level` defaults to `0` (free) when not supplied by the caller
  in live inference.
- All other counts default to 0 when the corresponding event class is absent.

## Model Architecture

- **Algorithm**: XGBoost classifier (binary:logistic)
- **n_estimators**: 400 (early stopping at round 36 best iter)
- **max_depth**: 6
- **learning_rate**: 0.1
- **subsample / colsample_bytree**: 0.9 / 0.9
- **min_child_weight**: 2
- **reg_lambda**: 1.0
- **scale_pos_weight**: ~3.53 (`neg_count / pos_count` from train fold)
- **eval_metric**: `auc`, with `early_stopping_rounds=20` on the test split
- **random_state**: 42

Features are scaled with `StandardScaler` (fit on train only, then applied to
test). Trees are scale-invariant in principle but we keep the scaler in the
pipeline so the same artifact serves any future linear or distance-based head
without re-engineering.

## Evaluation Metrics (Test Set, n=90)

| Metric | Value |
|---|---|
| AUC-ROC   | **0.9271** |
| Precision | 0.7619 |
| Recall    | 0.8000 |
| F1-Score  | 0.7805 |

Confusion Matrix:

|                | Predicted Retained | Predicted Churn |
|---|---|---|
| **Actual Retained** | 65 (TN) | 5 (FP)  |
| **Actual Churn**    | 4  (FN) | 16 (TP) |

Target was AUC-ROC ≥ 0.82; achieved 0.9271.

## Top SHAP Features (mean |SHAP| on test set)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `days_since_last_activity` | 1.7015 |
| 2 | `songs_played_total`       | 0.5260 |
| 3 | `thumbs_down_count`        | 0.2916 |
| 4 | `thumbs_up_count`          | 0.2818 |
| 5 | `add_to_playlist_count`    | 0.2239 |
| 6 | `avg_session_duration_min` | 0.1824 |
| 7 | `session_count_30d`        | 0.1720 |
| 8 | `subscription_level`       | 0.0167 |

Interpretation: recency dominates (~3x the next strongest signal). Engagement
depth (`songs_played_total`) is the strongest behavioral signal after recency.
`subscription_level` is near-noise on this dataset.

## Risk Score & Tiers

```python
risk_score = int(model.predict_proba(X)[:, 1] * 100)  # 0-100
```

| Tier | Range |
|---|---|
| low | 0-40 |
| medium | 41-70 |
| high | 71-85 |
| critical | 86-100 |

## Data Leakage Prevention

- Features are computed only from events that occur strictly before (or up to)
  the observation date — no post-cancellation lookahead.
- `StandardScaler` is fit on train only.
- Churn label is hard-coded from `Cancellation Confirmation` page presence and
  is not used as input to any feature.

## Known Failure Modes / Limitations

1. **Small distinct-user count.** The medium dump has only 448 distinct
   non-anonymous users. Even with stratification, the test fold is 90 users,
   so all metrics carry binomial confidence intervals on the order of ±8-10pp.
   Treat absolute metric values as point estimates.
2. **Single-vendor synthetic-style data.** Sparkify is a fictional service with
   uniform behavior distributions and no seasonality; real CSM dashboards will
   see drift not represented here.
3. **Class imbalance handled with `scale_pos_weight`, not resampling.** This
   keeps test set churn rates honest but biases the decision threshold; you
   may want to recalibrate `predict_proba` (e.g. isotonic) for downstream
   threshold-based alerting.
4. **`subscription_level` is near-noise** on Sparkify. Do not rely on it for
   plan-tier-specific behavior in production; the live `User.plan_type`
   distribution is much wider (free / starter / pro / enterprise).
5. **`session_count_30d` semantics differ between training and inference.**
   In training we have raw `sessionId`; in live Postgres we don't, so we
   reconstruct sessions by gap-clustering (>30 min idle = new session). This
   is well-correlated but not identical.
6. **No temporal cross-validation.** A user-disjoint stratified split is used,
   not a time-aware split. Concept drift is not measured by this evaluation.

## Model Artifacts

Saved under `ml/artifacts/`:

- `model.pkl` — joblib-serialized `XGBClassifier`
- `scaler.pkl` — joblib-serialized `StandardScaler` fit on train
- `feature_names.json` — list of 8 feature names (inference order)
- `correlation_report.json` — Claude / fallback churn-indicator analysis

## Inference API

`GET /users/{user_id}/risk` →

```json
{
  "user_id": "uuid",
  "risk_score": 75,
  "risk_tier": "high",
  "top_drivers": ["days_since_last_activity", "songs_played_total", "thumbs_down_count"],
  "shap_values": { "...": "..." },
  "model_version": "1.0"
}
```

---

**Model Version**: 1.0
**Training Date**: 2026-04-25
**Framework**: XGBoost 2.x + scikit-learn + SHAP + joblib
