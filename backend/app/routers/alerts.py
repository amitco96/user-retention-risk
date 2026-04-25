"""
Internal alerts endpoint for the nightly pipeline.

GET /pipeline/critical-users
- Returns the list of users currently in the `critical` risk tier.
- Bearer-token protected via API_PIPELINE_SECRET.
- Consumed by the n8n Slack-alert step.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth import verify_pipeline_token
from backend.app.db.models import RiskScore
from backend.app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


class CriticalUser(BaseModel):
    user_id: str
    risk_score: int
    reason: str
    recommended_action: str


class CriticalUsersResponse(BaseModel):
    users: List[CriticalUser]


@router.get(
    "/critical-users",
    response_model=CriticalUsersResponse,
    status_code=200,
    tags=["pipeline"],
    summary="List users currently in the critical risk tier",
    dependencies=[Depends(verify_pipeline_token)],
)
async def get_critical_users(
    db: AsyncSession = Depends(get_db),
) -> CriticalUsersResponse:
    try:
        stmt = (
            select(RiskScore)
            .where(RiskScore.risk_tier == "critical")
            .order_by(RiskScore.risk_score.desc())
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

        users = [
            CriticalUser(
                user_id=str(rs.user_id),
                risk_score=rs.risk_score,
                reason=rs.claude_reason or "",
                recommended_action=rs.claude_action or "",
            )
            for rs in rows
        ]
        return CriticalUsersResponse(users=users)

    except Exception as e:
        logger.error("Error fetching critical users: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
