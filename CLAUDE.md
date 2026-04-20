# CLAUDE.md — User Retention Risk Model

Production-grade ML system that scores user churn risk in real-time and explains WHY each user is at risk using Claude — deployed on AWS ECS Fargate.

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
│   ├── data_gen.py
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

**`data-engineer`** — Schema design, SQLAlchemy ORM models, data_gen.py, seed scripts. Does not touch ML, API, or infra.

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

## Data Schema

### `users`
`id` (UUID PK), `email`, `plan_type` (free/starter/pro/enterprise), `signup_date`, `created_at`, `updated_at`

### `events`
`id` (UUID PK), `user_id` (FK), `event_type` (login/feature_used/support_ticket), `event_metadata` (JSONB), `occurred_at`
Indexes: `(user_id)`, `(occurred_at)`, `(user_id, occurred_at DESC)`

### `risk_scores`
`id` (UUID PK), `user_id` (FK), `risk_score` (int 0–100), `risk_tier` (low/medium/high/critical), `top_drivers` (text[]), `shap_values` (JSONB), `claude_reason` (text), `claude_action` (text), `model_version`, `scored_at`
Indexes: `(user_id)`, `(scored_at DESC)`, `(risk_score DESC)`

---

## ML Features

```python
FEATURES = [
    "days_since_last_login",   # recency
    "session_count_30d",       # frequency
    "feature_usage_count",     # depth
    "support_tickets_open",    # friction
    "plan_type_encoded",       # value tier (free=0 → enterprise=3)
    "avg_session_duration_min",# quality
    "days_since_signup",       # tenure
    "login_streak_broken",     # binary: streak broken in last 7d
]
```

Churn label: `days_since_last_login >= 30 AND sessions_60d < 3`
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
- [ ] All phase deliverables implemented
- [ ] `pytest --cov=backend/app` passes at ≥ 80% coverage
- [ ] `/review` returns no critical issues
- [ ] Feature works end-to-end in docker-compose
- [ ] `git grep -r "sk-ant"` returns nothing