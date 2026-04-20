import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
    ARRAY,
    Index,
    Uuid,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User account with plan information."""

    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    plan_type = Column(
        Enum("free", "starter", "pro", "enterprise", name="plan_type_enum"),
        nullable=False,
        default="free",
    )
    signup_date = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    events = relationship("Event", back_populates="user", cascade="all, delete-orphan")
    risk_scores = relationship(
        "RiskScore", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, plan_type={self.plan_type})>"


class Event(Base):
    """User events (logins, feature usage, support tickets)."""

    __tablename__ = "events"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    event_type = Column(
        Enum(
            "login",
            "feature_used",
            "support_ticket",
            name="event_type_enum",
        ),
        nullable=False,
    )
    event_metadata = Column(JSON, nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    user = relationship("User", back_populates="events")

    # Indexes
    __table_args__ = (
        Index("ix_events_user_id_occurred_at", "user_id", "occurred_at", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<Event(id={self.id}, user_id={self.user_id}, event_type={self.event_type})>"


class RiskScore(Base):
    """Computed churn risk scores and explanations."""

    __tablename__ = "risk_scores"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    risk_score = Column(Integer, nullable=False)  # 0-100
    risk_tier = Column(
        Enum("low", "medium", "high", "critical", name="risk_tier_enum"),
        nullable=False,
    )
    top_drivers = Column(ARRAY(String), nullable=True)  # List of feature names
    shap_values = Column(JSON, nullable=True)  # SHAP values as JSON
    claude_reason = Column(String, nullable=True)  # Claude-generated explanation
    claude_action = Column(String, nullable=True)  # Claude-generated recommendation
    model_version = Column(String(50), nullable=False)
    scored_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    user = relationship("User", back_populates="risk_scores")

    # Indexes
    __table_args__ = (
        Index("ix_risk_scores_risk_score_desc", "risk_score", postgresql_using="btree"),
    )

    def __repr__(self):
        return f"<RiskScore(id={self.id}, user_id={self.user_id}, risk_score={self.risk_score})>"
