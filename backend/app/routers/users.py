"""
User risk scoring endpoints.

Endpoints:
- GET  /users/{user_id}/risk → RiskResponse
- GET  /users/at-risk?threshold=70 → List[RiskSummary]
- POST /users/{user_id}/risk/feedback → {action, notes}

Rate limited to 100 req/min per IP on /risk endpoint.
"""

import logging
import uuid
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.db.session import get_db
from backend.app.db.models import User, Event, RiskScore
from backend.app.schemas.risk import RiskResponse, RiskSummary
from backend.app.ml.model import predict
from backend.app.ml.explainer import explain_risk
from backend.app.ml.features import extract_user_features

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{user_id}/risk",
    response_model=RiskResponse,
    status_code=200,
    tags=["users"],
    summary="Get user churn risk score and explanation",
)
async def get_user_risk(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> RiskResponse:
    """
    Fetch user from DB, engineer features, score with model, explain with Claude.

    Args:
        user_id: User UUID
        db: Async database session

    Returns:
        RiskResponse with risk_score, risk_tier, reason, recommended_action

    Raises:
        HTTPException(404): User not found
        HTTPException(422): Invalid user_id format
        HTTPException(500): Model error or API error
    """
    try:
        # Convert user_id string to UUID
        try:
            user_uuid = uuid.UUID(user_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid user ID format")

        # Fetch user from database
        stmt = select(User).where(User.id == user_uuid)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch user's events
        stmt = select(Event).where(Event.user_id == user_uuid).order_by(Event.occurred_at)
        result = await db.execute(stmt)
        events = result.scalars().all()

        if not events:
            raise HTTPException(
                status_code=422,
                detail="No events found for user. Cannot score.",
            )

        # Extract features from events (pass Event objects directly)
        features = extract_user_features(events)

        # Score with model
        prediction = predict(user_id, features)

        # Compute user context for Claude
        now = datetime.utcnow()
        days_since_signup = (now - user.signup_date).days
        last_event_date = events[-1].occurred_at if events else user.signup_date
        days_since_last_login = (now - last_event_date).days

        user_context = {
            "plan_type": user.plan_type,
            "days_since_signup": days_since_signup,
            "days_since_last_login": days_since_last_login,
        }

        # Get Claude explanation
        explanation = await explain_risk(
            user_context=user_context,
            risk_score=prediction.risk_score,
            top_drivers=prediction.top_drivers,
            shap_values=prediction.shap_values,
        )

        # Build response
        response = RiskResponse(
            user_id=user_id,
            risk_score=prediction.risk_score,
            risk_tier=prediction.risk_tier,
            top_drivers=prediction.top_drivers,
            reason=explanation.reason,
            recommended_action=explanation.action,
            scored_at=datetime.utcnow(),
            model_version=prediction.model_version,
        )

        # Save to risk_scores table (optional: async save)
        risk_score_record = RiskScore(
            user_id=user_uuid,
            risk_score=prediction.risk_score,
            risk_tier=prediction.risk_tier,
            top_drivers=json.dumps(prediction.top_drivers),
            shap_values=prediction.shap_values,
            claude_reason=explanation.reason,
            claude_action=explanation.action,
            model_version=prediction.model_version,
            scored_at=response.scored_at,
        )
        db.add(risk_score_record)
        await db.commit()

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error for user {user_id}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error scoring user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/at-risk",
    response_model=List[RiskSummary],
    status_code=200,
    tags=["users"],
    summary="Get all users at or above risk threshold",
)
async def get_at_risk_users(
    threshold: int = Query(70, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[RiskSummary]:
    """
    Query risk_scores table for users at or above threshold.

    Args:
        threshold: Minimum risk score (0-100, default 70)
        db: Async database session

    Returns:
        List[RiskSummary] sorted by risk_score descending

    Raises:
        HTTPException(422): Invalid threshold
        HTTPException(500): Database error
    """
    try:
        # Query risk_scores table
        stmt = (
            select(RiskScore)
            .where(RiskScore.risk_score >= threshold)
            .order_by(RiskScore.risk_score.desc())
        )
        result = await db.execute(stmt)
        risk_scores = result.scalars().all()

        # Convert to RiskSummary
        summaries = [
            RiskSummary(
                user_id=str(rs.user_id),
                risk_score=rs.risk_score,
                risk_tier=rs.risk_tier,
                reason=rs.claude_reason or "",
            )
            for rs in risk_scores
        ]

        return summaries

    except Exception as e:
        logger.error(f"Error fetching at-risk users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{user_id}/risk/feedback",
    status_code=200,
    tags=["users"],
    summary="Record CSM feedback on risk score",
)
async def record_risk_feedback(
    user_id: str,
    action: str,
    notes: str = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Save CSM action and notes to risk_scores table.

    Args:
        user_id: User UUID
        action: CSM action taken (e.g., "email_sent", "call_scheduled")
        notes: Optional free-form notes
        db: Async database session

    Returns:
        Success message

    Raises:
        HTTPException(404): No risk score found for user
        HTTPException(500): Database error
    """
    try:
        # Convert user_id string to UUID
        try:
            user_uuid = uuid.UUID(user_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="Invalid user ID format")

        # Find most recent risk score for user
        stmt = (
            select(RiskScore)
            .where(RiskScore.user_id == user_uuid)
            .order_by(RiskScore.scored_at.desc())
        )
        result = await db.execute(stmt)
        risk_score = result.scalar_one_or_none()

        if risk_score is None:
            raise HTTPException(status_code=404, detail="No risk score found for user")

        # Store action (append to claude_action if it exists)
        if risk_score.claude_action:
            risk_score.claude_action = f"{risk_score.claude_action} [CSM: {action}]"
        else:
            risk_score.claude_action = action

        await db.commit()

        return {
            "status": "success",
            "message": f"Recorded action '{action}' for user {user_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording feedback for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
