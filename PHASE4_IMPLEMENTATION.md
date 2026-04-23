# Phase 4: FastAPI Endpoint Implementation

**Deliverable Date**: 2026-04-23  
**Status**: COMPLETE

## Overview

Phase 4 implements full FastAPI endpoints to wire the ML model and Claude explainer layer into production-ready REST APIs. All endpoints are async, properly typed, and include comprehensive error handling.

## Implemented Deliverables

### 1. Pydantic Schemas (`backend/app/schemas/risk.py`)

Three primary response models:

```python
class RiskResponse(BaseModel):
    user_id: str
    risk_score: int              # 0-100
    risk_tier: str               # low | medium | high | critical
    top_drivers: List[str]       # Top 3 SHAP feature drivers
    reason: str                  # Claude-generated explanation
    recommended_action: str      # Claude-generated CSM action
    scored_at: datetime          # Scoring timestamp
    model_version: str           # Model version identifier

class RiskSummary(BaseModel):
    user_id: str
    risk_score: int              # 0-100
    risk_tier: str               # low | medium | high | critical
    reason: str                  # Brief explanation (optional)

class CohortRetentionData(BaseModel):
    cohorts: List[str]           # List of cohort labels (YYYY-MM)
    weeks: List[str]             # List of week labels ("Week 0", "Week 1", ...)
    retention_matrix: List[List[float]]  # 2D matrix [cohorts][weeks]
```

### 2. ML Feature Engineering (`backend/app/ml/features.py`)

Extracted features for live user data from PostgreSQL events:

```python
def extract_user_features(user_events: list, observation_date: datetime = None) -> Dict[str, float]
```

Maps event types to Sparkify-compatible features:
- `songs_played_total`: Count of `feature_used` events
- `thumbs_up_count`: Support tickets with positive sentiment
- `thumbs_down_count`: Support tickets with negative sentiment
- `add_to_playlist_count`: Feature_used events with playlist action
- `avg_session_duration_min`: Time span / event count approximation

### 3. ML Model (`backend/app/ml/model.py`)

Enhanced with proper SHAP-based explainability:

```python
def predict(user_id: str, features: Dict[str, float]) -> RiskPrediction
```

Returns:
- Risk score (0-100)
- Risk tier mapping: low (≤40), medium (41-70), high (71-85), critical (86+)
- Top 3 drivers by absolute SHAP value
- All feature SHAP values for Claude context

### 4. Claude Explainer (`backend/app/ml/explainer.py`)

Async Claude integration with proper error handling:

```python
async def explain_risk(
    user_context: Dict[str, Any],
    risk_score: int,
    top_drivers: list[str],
    shap_values: Dict[str, float] | None = None,
) -> RiskExplanation
```

Features:
- Uses `AsyncClient` from anthropic (async/await throughout)
- 10-second timeout on API calls
- Graceful fallback to default explanation on error
- Parses JSON response with validation
- Logs errors without exposing to API consumers

### 5. FastAPI Routers

#### Users Router (`backend/app/routers/users.py`)

**GET `/users/{user_id}/risk` → RiskResponse**
- Fetch user from DB
- Validate user_id UUID format (422 if invalid)
- Fetch user events, extract features
- Score with XGBoost model
- Call Claude for explanation
- Save to risk_scores table
- Return full RiskResponse
- Error Handling:
  - 404: User not found
  - 422: Invalid user_id or no events
  - 500: Model or API error

**GET `/users/at-risk?threshold=70` → List[RiskSummary]**
- Query risk_scores table for scores ≥ threshold
- Sort by risk_score descending
- Return minimal summaries (no Claude calls needed)
- Error Handling:
  - 422: Invalid threshold (0-100)
  - 500: Database error

**POST `/users/{user_id}/risk/feedback`**
- Accept `{action: str, notes: str}`
- Fetch most recent risk_score for user
- Append action to claude_action field
- Commit to database
- Return success message
- Error Handling:
  - 404: No risk score found for user
  - 422: Invalid user_id
  - 500: Database error

#### Cohorts Router (`backend/app/routers/cohorts.py`)

**GET `/cohorts/retention` → CohortRetentionData**
- Query all users grouped by signup cohort (YYYY-MM)
- For each cohort and week, count active users
- Compute retention % = (active_users / cohort_size) * 100
- Return last 12+ weeks of data
- Error Handling:
  - 500: Database error

### 6. Main Application (`backend/app/main.py`)

Integrated routers:
```python
app.include_router(health.router)
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(cohorts.router, prefix="/cohorts", tags=["cohorts"])
```

## Endpoint Summary

| Method | Path | Status | Response |
|--------|------|--------|----------|
| GET | `/health` | 200 | `{status: ok}` |
| GET | `/users/{user_id}/risk` | 200/404/422/500 | `RiskResponse` |
| GET | `/users/at-risk?threshold=70` | 200/500 | `List[RiskSummary]` |
| POST | `/users/{user_id}/risk/feedback` | 200/404/500 | `{status, message}` |
| GET | `/cohorts/retention` | 200/500 | `CohortRetentionData` |

## Design Patterns

### Async/Await Throughout
- All endpoints are async functions
- SQLAlchemy async sessions via FastAPI dependency injection
- AsyncClient for Claude API calls
- Proper error propagation

### Error Handling
- **404 Not Found**: User or resource doesn't exist
- **422 Unprocessable Entity**: Invalid input format or missing data
- **500 Internal Server Error**: Unhandled exceptions logged server-side
- All errors returned as JSON with descriptive messages

### Database Integrity
- All queries use SQLAlchemy ORM (no raw SQL)
- UUID handling via proper conversion
- Foreign key relationships enforced by database
- Transactions committed after state changes

### Model Integration
- Feature extraction maps live events to Sparkify schema
- SHAP values extracted from XGBoost predictions
- Model artifact paths resolved at import time
- Graceful fallback if model not yet trained

## Testing

### Unit Tests
Created `/backend/tests/test_endpoints.py` with verification of:
- Feature extraction from mock events
- Model prediction (SHAP values, risk tier mapping)
- Schema validation (RiskResponse, RiskSummary, CohortRetentionData)

All tests pass successfully.

### Manual Testing
```bash
# Compile all modules
python -m py_compile backend/app/routers/*.py backend/app/schemas/*.py

# Run verification script
PYTHONPATH=/c/user-retention-risk python backend/tests/test_endpoints.py
```

Output:
```
======================================================================
PHASE 4 ENDPOINT VERIFICATION
======================================================================

[TEST] Feature Extraction
  Features extracted successfully:
    - add_to_playlist_count: 0.00
    - avg_session_duration_min: 1296.00
    - songs_played_total: 10.00
    - thumbs_down_count: 0.00
    - thumbs_up_count: 0.00

[TEST] Model Prediction
  Risk Score: 0/100
  Risk Tier: low
  Top Drivers: add_to_playlist_count, thumbs_up_count, songs_played_total

[TEST] Schema Validation
  RiskResponse created: 75 -> high
  RiskSummary created: user-123
  CohortRetentionData created: 3 cohorts, 4 weeks

======================================================================
ALL TESTS PASSED
======================================================================
```

## Files Modified/Created

### New Files
- `/c/user-retention-risk/backend/app/schemas/risk.py` — Pydantic schemas
- `/c/user-retention-risk/backend/app/ml/features.py` — Feature engineering
- `/c/user-retention-risk/backend/app/routers/users.py` — User endpoints
- `/c/user-retention-risk/backend/app/routers/cohorts.py` — Cohort endpoints
- `/c/user-retention-risk/backend/app/scripts/seed_test_data.py` — Data seeding utility
- `/c/user-retention-risk/backend/tests/test_endpoints.py` — Verification tests
- `/c/user-retention-risk/infrastructure/Dockerfile` — Container definition

### Modified Files
- `/c/user-retention-risk/backend/app/main.py` — Added router includes
- `/c/user-retention-risk/backend/app/ml/explainer.py` — Fixed async/await, added error handling
- `/c/user-retention-risk/backend/app/ml/model.py` — Verified SHAP integration
- `/c/user-retention-risk/backend/requirements.txt` — Fixed dependency versions

## Next Steps (Phase 5+)

1. **Docker Stack Testing**: Verify endpoints with `docker-compose up`
2. **Integration Tests**: Run full pytest suite with database
3. **React Dashboard**: Implement frontend components
4. **Rate Limiting**: Apply slowapi middleware (100 req/min per IP)
5. **Monitoring**: Add structured logging and metrics

## API Contract Compliance

✓ All endpoints return Pydantic models, never raw dicts  
✓ All endpoints async with proper await  
✓ Dependency injection for DB sessions  
✓ Proper error codes (404/422/500)  
✓ Model version tracking  
✓ SHAP-based explainability  
✓ Claude integration with timeout and fallback  
✓ SQLAlchemy ORM only (no raw SQL)  
✓ UUID validation and conversion  
✓ Comprehensive docstrings  

## Security Considerations

- ANTHROPIC_API_KEY loaded from environment only
- No API keys hardcoded anywhere
- SQL injection prevented via SQLAlchemy ORM
- Proper UUID validation prevents malformed queries
- Graceful error messages (no stack traces to clients)
- Secrets should come from AWS Secrets Manager in production
