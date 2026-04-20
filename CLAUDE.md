# CLAUDE.md — User Retention Risk Model

Production-grade ML system that scores user churn risk in real-time and explains WHY each user is at risk using Claude — deployed on AWS ECS Fargate.

## Progress

**Phase 1: ✓ COMPLETE** (commit b72f5fd)
- SQLAlchemy 2.0 async ORM models (users, events, risk_scores) with PostgreSQL 16
- Async session factory with auto-create tables on startup
- Synthetic data generator: 1000 users, 85,638 events, 17% churn rate
- Docker stack: FastAPI /health, postgres service, docker-compose orchestration
- Commit: *feat: phase 1 — SQLAlchemy models + PostgreSQL data seeding*
- **Note**: switching from synthetic data to Sparkify dataset (Udacity mini, 128MB) — raw event logs from a music streaming service

**Phase 2: IN PROGRESS**
- Feature engineering (pandas + SQL feature extraction)
- XGBoost model training + SHAP explainability
- Claude integration for reason/action generation

---

## Architecture

```
PostgreSQL (user events)
         ↓
Feature Engineering (pandas + SQLAlchemy)
         ↓
XGBoost Churn Model → risk_score (0–100) + top_drivers[]
         ↓
Claude claude-sonnet-4-20250514 → reason + recommended_action
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

Engineered from raw Sparkify events in `ml/feature_engineering.py`:

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
- [x] **Phase 1**: All deliverables (models, session, data_gen, docker stack)
- [x] SQLAlchemy ORM models + PostgreSQL schema (/c/user-retention-risk/backend/app/db/)
- [x] Data seeding: 1000 users, 85k events, realistic cohorts
- [x] Docker stack operational (FastAPI + Postgres)
- [ ] Phase 2: Feature engineering + XGBoost model training
- [ ] `pytest --cov=backend/app` passes at ≥ 80% coverage
- [ ] `/review` returns no critical issues
- [ ] `git grep -r "sk-ant"` returns nothing