import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import health, users, cohorts
from backend.app.db.session import create_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="User Retention Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(cohorts.router, prefix="/cohorts", tags=["cohorts"])


@app.on_event("startup")
async def startup_event():
    logger.info("API started")
    await create_tables()
    logger.info("Database tables initialized")
