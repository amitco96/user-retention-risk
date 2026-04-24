"""
Pydantic schemas for risk responses and cohort data.
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class RiskResponse(BaseModel):
    """Full risk score response with explanation."""

    user_id: str = Field(..., description="User identifier")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_tier: str = Field(..., description="low | medium | high | critical")
    top_drivers: List[str] = Field(default_factory=list, description="Top 3 SHAP feature drivers")
    reason: str = Field(..., description="Claude-generated explanation of risk")
    recommended_action: str = Field(..., description="Claude-generated recommended CSM action")
    scored_at: datetime = Field(..., description="Timestamp of scoring")
    model_version: str = Field(default="1.0", description="Model version")


class RiskSummary(BaseModel):
    """Minimal risk score for list responses."""

    user_id: str = Field(..., description="User identifier")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_tier: str = Field(..., description="low | medium | high | critical")
    reason: str = Field(default="", description="Brief reason for risk")


class CohortWeekData(BaseModel):
    """Retention data for a single cohort week."""

    week: int
    retention_pct: float


class Cohort(BaseModel):
    """A signup cohort with week-over-week retention."""

    cohort_week: str
    weeks: List[CohortWeekData]


class CohortRetentionData(BaseModel):
    """Cohort-based retention analysis."""

    cohorts: List[Cohort] = Field(default_factory=list)
