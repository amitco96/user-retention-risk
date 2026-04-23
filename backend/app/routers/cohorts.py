"""
Cohort-based retention analysis endpoints.

Endpoints:
- GET /cohorts/retention → CohortRetentionData
  Returns week-over-week retention percentages for signup cohorts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Set

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.app.db.session import get_db
from backend.app.db.models import User, Event
from backend.app.schemas.risk import CohortRetentionData

logger = logging.getLogger(__name__)

router = APIRouter()


def get_week_number(date: datetime, cohort_start: datetime) -> int:
    """
    Calculate week number relative to cohort start date.

    Args:
        date: Reference date
        cohort_start: Cohort start date (signup date)

    Returns:
        Week number (0, 1, 2, ...)
    """
    delta = date - cohort_start
    return delta.days // 7


@router.get(
    "/retention",
    response_model=CohortRetentionData,
    status_code=200,
    tags=["cohorts"],
    summary="Get cohort-based retention matrix",
)
async def get_cohort_retention(
    db: AsyncSession = Depends(get_db),
) -> CohortRetentionData:
    """
    Query events and users to compute week-over-week retention by signup cohort.

    Retention for week N = (users active in week N) / (users in cohort) * 100

    Returns:
        CohortRetentionData with retention_matrix [cohorts][weeks]

    Raises:
        HTTPException(500): Database error
    """
    try:
        # Fetch all users with signup dates
        stmt = select(User).order_by(User.signup_date)
        result = await db.execute(stmt)
        users = result.scalars().all()

        if not users:
            return CohortRetentionData(
                cohorts=[],
                weeks=[],
                retention_matrix=[],
            )

        # Group users by cohort (week of signup)
        cohort_dict: Dict[str, list] = {}
        for user in users:
            # Cohort label: YYYY-W## (e.g., "2026-01" for January 2026)
            cohort_label = user.signup_date.strftime("%Y-%m")
            if cohort_label not in cohort_dict:
                cohort_dict[cohort_label] = []
            cohort_dict[cohort_label].append(user)

        cohort_labels = sorted(cohort_dict.keys())

        # Fetch all events
        stmt = select(Event).order_by(Event.occurred_at)
        result = await db.execute(stmt)
        all_events = result.scalars().all()

        if not all_events:
            return CohortRetentionData(
                cohorts=cohort_labels,
                weeks=[],
                retention_matrix=[[]] * len(cohort_labels),
            )

        # Determine the max week from the latest event
        max_date = all_events[-1].occurred_at
        min_cohort_date = min(u.signup_date for u in users)
        max_weeks = get_week_number(max_date, min_cohort_date) + 1

        # Build retention matrix
        retention_matrix = []
        for cohort_label in cohort_labels:
            cohort_users = cohort_dict[cohort_label]
            cohort_start = cohort_users[0].signup_date

            # Get user IDs in this cohort
            cohort_user_ids = set(u.id for u in cohort_users)

            # For each week, count active users
            cohort_retention = []
            for week_num in range(max_weeks + 1):
                week_start = cohort_start + timedelta(weeks=week_num)
                week_end = week_start + timedelta(weeks=1)

                # Count users from this cohort with events in this week
                active_users = set()
                for event in all_events:
                    if (
                        event.user_id in cohort_user_ids
                        and week_start <= event.occurred_at < week_end
                    ):
                        active_users.add(event.user_id)

                # Retention = active / total in cohort * 100
                if len(cohort_users) > 0:
                    retention_pct = (len(active_users) / len(cohort_users)) * 100
                else:
                    retention_pct = 0.0

                cohort_retention.append(retention_pct)

            retention_matrix.append(cohort_retention)

        # Generate week labels
        week_labels = [f"Week {i}" for i in range(max_weeks + 1)]

        return CohortRetentionData(
            cohorts=cohort_labels,
            weeks=week_labels,
            retention_matrix=retention_matrix,
        )

    except Exception as e:
        logger.error(f"Error computing cohort retention: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
