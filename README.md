# User Retention Risk Model

Production ML system that scores user churn risk in real time, explains every prediction with Claude, and pushes nightly alerts to the retention team.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EB6C2D)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204-D97757)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

## What This Is

An end-to-end churn-risk system that replaces the static "retention dashboard nobody opens" with a living pipeline: events land in Postgres, a model rescores every user nightly, Claude generates a one-sentence reason and recommended action per critical user, and Slack pings the CSM team. The work an analytics engineer would actually ship — not a notebook with a confusion matrix.

## Why It's Different

- **Claude picks the features, not me.** Before training, `ml/ai_analysis.py` asks Claude to analyze the raw event data and rank leading indicators of churn. The model trains on what Claude flagged, not on a hardcoded list copied from a tutorial.
- **n8n runs the pipeline on a schedule.** Nightly rescore of every user, write to `risk_scores`, post to Slack for anything tier=critical. No human in the loop; no `python score.py` ritual.
- **Alerts push, dashboards pull.** The CSM team gets a Slack message when a paying user crosses the risk threshold. They don't have to remember to check anything.
- **Every score has a reason.** XGBoost + SHAP surface the top drivers, Claude turns them into one sentence of plain English plus a concrete next step.
- **97% test coverage, 148 tests, async all the way down.** FastAPI async, SQLAlchemy 2.0 async ORM, async Claude client with timeout + fallback, dialect-aware cohort SQL.

## Architecture

```
Sparkify event logs
        |
        v
Claude correlation analysis  -->  ml/artifacts/correlation_report.json
        |
        v
XGBoost training (SHAP)      -->  model.pkl + feature_names.json
        |
        v
PostgreSQL live events  <-- ingestion
        |
        v
n8n nightly pipeline:  feature engineering --> score all users --> upsert risk_scores --> Slack alert (critical tier)
        |
        v
FastAPI async REST  <-->  React dashboard (risk table, user card, cohort chart)
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115+ async, Python 3.11, slowapi rate limiting |
| ML | XGBoost 2.x, SHAP, scikit-learn |
| AI | Claude `claude-sonnet-4-20250514` — correlation analysis + per-user explanations |
| Database | PostgreSQL 16, SQLAlchemy 2.0 async ORM |
| Frontend | React 18, Recharts, Tailwind CSS, Vite |
| Automation | n8n nightly pipeline, Slack webhook |
| Infrastructure | Docker + Compose, ECS Fargate task definition, GitHub Actions CI/CD |
| Testing | pytest, pytest-asyncio, 97% coverage, 148 tests |

## Dataset

Sparkify (Udacity) — real event logs from a music-streaming service. 448 users, 528k events, 22% churn. Churn label: any user who hit the `Cancellation Confirmation` page.

Model performance on held-out test set: **AUC-ROC 0.9271, F1 0.78.**

## Project Structure

```
user-retention-risk/
├── backend/            FastAPI app — routers, ML wrapper, Claude explainer, async DB
├── ml/                 Feature engineering, training, AI correlation analysis, artifacts
├── frontend/           React dashboard — risk table, user card, cohort chart
├── infrastructure/     Dockerfile, docker-compose, ECS task definition
├── n8n/                Nightly rescore workflow + Slack alert
├── .github/workflows/  CI (lint + test + coverage), deploy (ECR + ECS)
└── CLAUDE.md           Architecture + phase-by-phase build log
```

## Quick Start

```bash
git clone <repo-url> && cd user-retention-risk
cp .env.example .env                                       # fill in ANTHROPIC_API_KEY
docker compose -f infrastructure/docker-compose.yml up -d
docker exec retention-app sh -c "cd /app && PYTHONPATH=/app python backend/seed_real_users.py"
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/users/{id}/risk` | Score + Claude explanation for one user |
| GET | `/users/at-risk?threshold=70` | All users above the risk threshold |
| GET | `/cohorts/retention` | Week-over-week retention by signup cohort |
| POST | `/users/{id}/risk/feedback` | CSM feedback on a prediction |
| POST | `/pipeline/rescore-all` | Nightly rescore (bearer-token, n8n) |
| GET | `/pipeline/critical-users` | Critical-tier users for alerting (bearer-token) |

## Key Engineering Decisions

- **Claude analyzes the dataset *before* training, not just after.** Most "AI-powered" ML systems use an LLM to explain a model trained on hardcoded features. Here Claude reads the raw event distribution and recommends which signals are leading indicators of churn — the feature list is data-derived, not lifted from a blog post.
- **n8n over cron.** A workflow tool gives observability (every run is logged with input/output), retries, and a UI a non-engineer can read. A `cron` line in `/etc/crontab` gives none of that.
- **Tests call async router functions directly, not via ASGI transport.** A Python 3.11 + `coverage.py` interaction caused HTTP-transport tests to under-report coverage on async paths. Direct calls (`TestRouterLogicDirect`) get the same code path with accurate tracking — same correctness, real numbers.
- **Sparkify timestamps were re-anchored to "now".** The raw dataset is from 2018; recency features (`days_since_last_activity`) only make sense relative to a current clock. Re-anchoring on ingest keeps the model's recency signal alive in a live system.

## Running Tests

```bash
docker exec retention-app pytest
```

Expected: `148 passed, 0 failures`, coverage ≥ 97% on `backend/app`.

## Production Deploy

The deployment pipeline is fully implemented and was verified end-to-end on AWS:
- Multi-stage Docker image (ECR)
- IAM least-privilege roles (execution role + task role)
- Secrets Manager for `ANTHROPIC_KEY`, `DATABASE_URL`, `API_PIPELINE_SECRET`
- ECS Fargate service (512 CPU / 1024 MB) with CloudWatch logging
- GitHub Actions: lint → test → ECR push → `update-service` → wait stable

To bring it live: provision RDS in the same VPC, update the `DATABASE_URL` 
secret, and push to main. The pipeline does the rest.