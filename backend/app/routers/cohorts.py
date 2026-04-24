"""
Cohort-based retention analysis endpoints.

Endpoints:
- GET /cohorts/retention → CohortRetentionData
  Returns week-over-week retention percentages for signup cohorts.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.app.db.session import get_db
from backend.app.schemas.risk import Cohort, CohortRetentionData, CohortWeekData

logger = logging.getLogger(__name__)

router = APIRouter()


COHORT_RETENTION_SQL = text("""
WITH cohort_sizes AS (
    SELECT
        to_char(date_trunc('month', signup_date), 'YYYY-MM') AS cohort_week,
        COUNT(*) AS cohort_size,
        MIN(signup_date) AS cohort_start
    FROM users
    GROUP BY cohort_week
),
activity AS (
    SELECT
        to_char(date_trunc('month', u.signup_date), 'YYYY-MM') AS cohort_week,
        GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (e.occurred_at - u.signup_date)) / 604800))::int AS week,
        COUNT(DISTINCT e.user_id) AS active
    FROM users u
    JOIN events e ON e.user_id = u.id AND e.occurred_at >= u.signup_date
    GROUP BY cohort_week, week
)
SELECT
    a.cohort_week,
    a.week,
    ROUND((a.active::numeric / c.cohort_size) * 100, 1) AS retention_pct
FROM activity a
JOIN cohort_sizes c ON c.cohort_week = a.cohort_week
WHERE a.week <= 12
ORDER BY a.cohort_week, a.week
""")


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
    try:
        result = await db.execute(COHORT_RETENTION_SQL)
        rows = result.all()

        by_cohort: dict[str, list[CohortWeekData]] = {}
        for row in rows:
            by_cohort.setdefault(row.cohort_week, []).append(
                CohortWeekData(week=int(row.week), retention_pct=float(row.retention_pct))
            )

        cohorts = [
            Cohort(cohort_week=label, weeks=weeks)
            for label, weeks in sorted(by_cohort.items())
        ]

        return CohortRetentionData(cohorts=cohorts)

    except Exception as e:
        logger.error(f"Error computing cohort retention: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
