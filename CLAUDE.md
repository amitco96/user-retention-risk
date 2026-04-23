# CLAUDE.md — User Retention Risk Model

Production-grade ML system that scores user churn risk in real-time and explains WHY each user is at risk using Claude — deployed on AWS ECS Fargate.

## Progress

**Phase 1: ✓ COMPLETE** (commit b72f5fd)
- SQLAlchemy 2.0 async ORM models (users, events, risk_scores) with PostgreSQL 16
- Async session factory with auto-create tables on startup
- Synthetic data generator: 1000 users, 85,638 events, 17% churn rate
- Docker stack: FastAPI /health, postgres service, docker-compose orchestration
- Commit: *feat: phase 1 — SQLAlchemy models + PostgreSQL data seeding*

**Phase 2: ✓ COMPLETE**
- XGBoost trained on Sparkify dataset, AUC-ROC 0.9861
- Feature engineering from raw Sparkify events
- SHAP explainability integrated

**Phase 2b: IN PROGRESS**
- Claude correlation analysis of Sparkify dataset
- ml/ai_analysis.py outputs correlation_report.json
- Claude recommends leading churn indicators

**Phase 4: ✓ COMPLETE**
- FastAPI endpoints: GET /users/{user_id}/risk, GET /users/at-risk?threshold, POST /users/{user_id}/risk/feedback
- GET /cohorts/retention — cohort-based retention analysis
- Pydantic schemas: RiskResponse, RiskSummary, CohortRetentionData
- ML model wrapper with SHAP explainability
- Claude async explainer (Sonnet 4 with 10s timeout + fallback)
- Feature engineering pipeline for live PostgreSQL events
- Docker stack tested & operational (port 8000)
- All endpoints returning valid JSON responses ✓

---

## Architecture

```
Sparkify dataset (training only)
         ↓
Claude analyzes dataset correlations → outputs correlation_report.json
         ↓
XGBoost trained on Claude-recommended features → model.pkl + scaler.pkl + feature_names.json
         ↓
PostgreSQL live user events
         ↓
n8n nightly pipeline: feature engineering → score all users → save to risk_scores table → Slack alert for critical users
         ↓
FastAPI REST API ←→ React Dashboard
         ↓
AWS ECS Fargate ← GitHub Actions CI/CD
```

---

## Repo Structure

```
user-retention-risk/
├── CLAUDE.md
├── .claude/
│   ├── agents/
│   └── commands/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── users.py
│   │   │   ├── cohorts.py
│   │   │   └── health.py
│   │   ├── ml/
│   │   │   ├── model.py
│   │   │   ├── features.py
│   │   │   └── explainer.py
│   │   ├── db/
│   │   │   ├── session.py
│   │   │   └── models.py
│   │   └── schemas/
│   │       └── risk.py
│   ├── tests/
│   └── requirements.txt
├── ml/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── model_card.md
│   └── artifacts/
│       ├── model.pkl
│       └── feature_names.json
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── RiskTable.jsx
│   │   │   ├── UserRiskCard.jsx
│   │   │   └── CohortChart.jsx
│   │   └── api/client.js
│   └── package.json
├── infrastructure/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── ecs/
│       └── task-definition.json
└── .github/
    └── workflows/
        ├── ci.yml
        └── deploy.yml
```

---

## Agent Roster

**`data-engineer`** — Schema design, SQLAlchemy ORM models, data_loader.py, seed scripts. Does not touch ML, API, or infra.

**`ml-engineer`** — Feature engineering, XGBoost training, SHAP analysis, model serialization, model_card.md. Does not touch API or infra.

**`api-engineer`** — FastAPI routers, Pydantic schemas, Claude explainer layer, React dashboard. Does not touch ML training or infra.

**`devops-engineer`** — Dockerfile, docker-compose, ECS task definition, GitHub Actions CI/CD. Does not touch app code.

**`code-reviewer`** — Read-only. Reviews for security, type safety, error handling, test coverage. Produces structured pass/fail report.

**`test-engineer`** — Writes and runs pytest tests. Reports coverage gaps.

---

## MCP Servers

```json
{
  "mcpServers": {
    "github":     { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] },
    "postgres":   { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost:5432/retention_dev"] },
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
  }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115+ async |
| ML | XGBoost 2.x + scikit-learn |
| Explainability | SHAP + Claude claude-sonnet-4-20250514 |
| Database | PostgreSQL 16, SQLAlchemy 2.0 async ORM |
| Frontend | React 18 + Recharts + Tailwind |
| Orchestration | n8n — Nightly rescore + Slack alerts |
| Container | Docker + ECS Fargate (512 CPU / 1024 MB) |
| CI/CD | GitHub Actions |
| Secrets | AWS Secrets Manager |

---

## Dataset

**Sparkify (Udacity mini — 128MB)**
Raw event logs from a fictitious music streaming service.
Download: search "Sparkify dataset" on Kaggle or Udacity.
Place at: `ml/data/sparkify_mini.json`

Key raw columns used:
- `userId` — user identifier
- `sessionId` — groups events into sessions
- `page` — event type (NextSong, Thumbs Up, Add to Playlist, Roll Advert, Logout, Submit Downgrade, Cancellation Confirmation)
- `ts` — event timestamp (ms)
- `level` — subscription tier (free/paid)
- `registration` — account creation timestamp
- `length` — song duration (seconds)
- `gender`, `location` — user demographics

Churn label: user who visited "Cancellation Confirmation" page

---

## ML Features

Engineered from raw Sparkify events in `ml/feature_engineering.py`. **Features are not hardcoded.** `ml/ai_analysis.py` asks Claude to analyze the Sparkify dataset and recommend which behavioral signals are the strongest leading indicators of churn. Output saved to `ml/artifacts/correlation_report.json` and consumed by `train.py`.

```python
FEATURES = [
    "days_since_last_activity",  # recency
    "session_count_30d",         # frequency
    "songs_played_total",        # depth of engagement
    "thumbs_up_count",           # positive signal
    "thumbs_down_count",         # negative signal
    "add_to_playlist_count",     # stickiness signal
    "avg_session_duration_min",  # quality
    "subscription_level",        # free=0 / paid=1
]
```

Churn label: userId appears on "Cancellation Confirmation" page
Target: AUC-ROC ≥ 0.82

---

## API Contracts

### `RiskResponse`
```python
class RiskResponse(BaseModel):
    user_id: str
    risk_score: int            # 0-100
    risk_tier: str             # low | medium | high | critical
    top_drivers: List[str]
    reason: str                # Claude-generated
    recommended_action: str    # Claude-generated
    scored_at: datetime
    model_version: str
```

### Endpoints
```
GET  /health
GET  /users/{user_id}/risk            → RiskResponse
GET  /users/at-risk?threshold=70      → List[RiskSummary]
GET  /cohorts/retention               → CohortRetentionData
POST /users/{user_id}/risk/feedback
```

---

## Claude Explainer Prompt

```
You are a retention analyst. A user has a churn risk score of {score}/100.

Top risk drivers (from SHAP analysis):
{driver_1}: {direction} (weight: {weight})
{driver_2}: {direction}
{driver_3}: {direction}

User context:
- Plan: {plan_type}
- Tenure: {days_since_signup} days
- Last login: {days_since_last_login} days ago

In ONE sentence each:
1. reason: why is this user at risk?
2. action: what specific action should a CSM take this week?

Respond only in JSON: {"reason": "...", "action": "..."}
```

---

## Security Rules

- No hardcoded `ANTHROPIC_API_KEY`, `DATABASE_URL`, or AWS credentials anywhere
- Secrets via `os.environ` locally, AWS Secrets Manager in production
- Rate limit `/users/{id}/risk`: 100 req/min per IP (slowapi)
- SQL via SQLAlchemy ORM only — no raw f-string queries

---

## Definition of Done

A phase is complete when:
- [x] **Phase 1**: All deliverables (models, session, data_loader, docker stack)
- [x] SQLAlchemy ORM models + PostgreSQL schema (/c/user-retention-risk/backend/app/db/)
- [x] Data loading: Sparkify dataset ingested into PostgreSQL
- [x] Docker stack operational (FastAPI + Postgres)
- [x] Phase 2: Feature engineering + XGBoost model training
- [ ] Phase 2b: Claude correlation analysis step
- [x] **Phase 4**: FastAPI endpoints wired to model and explainer
  - [x] Pydantic schemas (RiskResponse, RiskSummary, CohortRetentionData)
  - [x] ML model wrapper + SHAP explainability
  - [x] Claude async explainer (10s timeout, graceful fallback)
  - [x] All 5 endpoints tested and returning valid JSON
  - [x] Feature engineering pipeline (extract_user_features)
  - [x] Cohort retention analysis (week-over-week %)
- [ ] Phase 5: `pytest --cov=backend/app` passes at ≥ 80% coverage
- [ ] `/review` returns no critical issues
- [ ] `git grep -r "sk-ant"` returns nothing

---

## Remaining Phases

**Phase 2b**: `ml/ai_analysis.py` — Claude analyzes Sparkify dataset, outputs `ml/artifacts/correlation_report.json`

**Phase 5**: `pytest` coverage ≥ 80% — unit & integration tests for all routers + ML pipeline

**Phase 6**: React dashboard — risk table, user card, cohort chart

**Phase 7**: n8n nightly pipeline and Slack alert — rescore all users, alert on critical risk

**Phase 8**: ECS Fargate production deploy — GitHub Actions CI/CD → AWS ECS