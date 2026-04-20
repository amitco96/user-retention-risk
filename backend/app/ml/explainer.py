import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class RiskExplanation:
    """Claude-generated explanation and recommended action for user churn risk."""
    reason: str
    action: str


async def explain_risk(
    user_context: Dict[str, Any],
    risk_score: int,
    top_drivers: list[str],
    shap_values: Dict[str, float] | None = None,
) -> RiskExplanation:
    """
    Generate human-readable explanation and recommended action for a user's churn risk
    using Claude Sonnet 4.

    Args:
        user_context: Dict with keys: plan_type, days_since_signup, days_since_last_login
        risk_score: Integer 0-100
        top_drivers: List of top 3 feature names
        shap_values: Optional dict mapping driver name to SHAP weight (for future use)

    Returns:
        RiskExplanation with reason and action fields

    Gracefully falls back to default response on API error or parse failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, returning default explanation")
        return RiskExplanation(
            reason="Unable to analyze.",
            action="Contact support."
        )

    # Build driver descriptions for the prompt
    driver_descriptions = []
    if shap_values:
        for i, driver in enumerate(top_drivers[:3], start=1):
            weight = shap_values.get(driver, 0.0)
            direction = "↑ increases risk" if weight > 0 else "↓ decreases risk"
            driver_descriptions.append(f"{driver}: {direction} (weight: {weight:.3f})")
    else:
        # Fallback if SHAP values not available
        for i, driver in enumerate(top_drivers[:3], start=1):
            driver_descriptions.append(f"{driver}: (from model)")

    # Ensure we have at least 3 drivers for the prompt
    while len(driver_descriptions) < 3:
        driver_descriptions.append("(no additional drivers)")

    # Extract user context with safe defaults
    plan_type = user_context.get("plan_type", "unknown")
    days_since_signup = user_context.get("days_since_signup", 0)
    days_since_last_login = user_context.get("days_since_last_login", 0)

    # Construct the prompt based on CLAUDE.md template
    prompt = f"""You are a retention analyst. A user has a churn risk score of {risk_score}/100.

Top risk drivers (from SHAP analysis):
{driver_descriptions[0]}
{driver_descriptions[1]}
{driver_descriptions[2]}

User context:
- Plan: {plan_type}
- Tenure: {days_since_signup} days
- Last login: {days_since_last_login} days ago

In ONE sentence each:
1. reason: why is this user at risk?
2. action: what specific action should a CSM take this week?

Respond only in JSON: {{"reason": "...", "action": "..."}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            timeout=10.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract text from response
        response_text = message.content[0].text

        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            reason = response_json.get("reason", "").strip()
            action = response_json.get("action", "").strip()

            if not reason or not action:
                logger.warning("Claude response missing reason or action, using fallback")
                return RiskExplanation(
                    reason="Unable to analyze.",
                    action="Contact support."
                )

            return RiskExplanation(reason=reason, action=action)

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Claude response as JSON: {response_text}")
            return RiskExplanation(
                reason="Unable to analyze.",
                action="Contact support."
            )

    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return RiskExplanation(
            reason="Unable to analyze.",
            action="Contact support."
        )
    except Exception as e:
        logger.error(f"Unexpected error in explain_risk: {e}")
        return RiskExplanation(
            reason="Unable to analyze.",
            action="Contact support."
        )
