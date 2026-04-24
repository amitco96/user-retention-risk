"""
Seed risk_scores table by scoring all users with the ML model.
Run inside the container: python backend/seed_risk_scores.py
"""
import asyncio
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from backend.app.db.session import engine, async_session_maker
from backend.app.db.models import User, Event, RiskScore, Base
from backend.app.ml.model import predict
from backend.app.ml.features import extract_user_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as db:
        result = await db.execute(select(User))
        users = result.scalars().all()
        logger.info(f"Scoring {len(users)} users...")

        scored = 0
        for user in users:
            try:
                ev_result = await db.execute(
                    select(Event).where(Event.user_id == user.id).order_by(Event.occurred_at)
                )
                events = ev_result.scalars().all()
                if not events:
                    continue

                features = extract_user_features(events)
                pred = predict(str(user.id), features)

                now = datetime.now(timezone.utc)
                days_signup = (now - user.signup_date).days
                last_event = events[-1].occurred_at
                days_inactive = (now - last_event).days

                reason = (
                    f"User inactive for {days_inactive}d; "
                    f"top driver: {pred.top_drivers[0] if pred.top_drivers else 'unknown'}"
                )
                action = (
                    "Send re-engagement email with personalized offer"
                    if pred.risk_score >= 70
                    else "Monitor activity over next 2 weeks"
                )

                db.add(RiskScore(
                    user_id=user.id,
                    risk_score=pred.risk_score,
                    risk_tier=pred.risk_tier,
                    top_drivers=json.dumps(pred.top_drivers),
                    shap_values=pred.shap_values,
                    claude_reason=reason,
                    claude_action=action,
                    model_version=pred.model_version,
                    scored_at=now,
                ))
                scored += 1

                if scored % 100 == 0:
                    await db.commit()
                    logger.info(f"  committed {scored}/{len(users)}")

            except Exception as e:
                logger.warning(f"Skipped user {user.id}: {e}")

        await db.commit()
        logger.info(f"Done. Scored {scored} users.")

        count = await db.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM risk_scores")
        )
        logger.info(f"risk_scores rows: {count.scalar()}")


if __name__ == "__main__":
    asyncio.run(seed())
