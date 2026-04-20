import logging
from fastapi import FastAPI
from backend.app.routers import health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="User Retention Risk API")

app.include_router(health.router)


@app.on_event("startup")
async def startup_event():
    logger.info("API started")
