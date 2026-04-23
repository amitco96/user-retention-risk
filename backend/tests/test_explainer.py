import json
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import anthropic

from backend.app.ml.explainer import explain_risk, RiskExplanation


@pytest.fixture
def sample_user_context():
    """Sample user context for testing."""
    return {
        "plan_type": "pro",
        "days_since_signup": 180,
        "days_since_last_login": 25,
    }


@pytest.fixture
def sample_risk_inputs():
    """Sample risk prediction inputs."""
    return {
        "risk_score": 78,
        "top_drivers": ["days_since_last_login", "session_count_30d", "support_tickets_open"],
        "shap_values": {
            "days_since_last_login": 0.45,
            "session_count_30d": -0.12,
            "support_tickets_open": 0.18,
        },
    }


class TestExplainerSuccess:
    """Test successful Claude API responses."""

    @pytest.mark.asyncio
    async def test_successful_explanation_with_shap_values(self, sample_user_context, sample_risk_inputs):
        """Test successful JSON response is parsed correctly."""
        expected_response = {
            "reason": "User hasn't logged in for 25 days and engagement is dropping.",
            "action": "Schedule a check-in call and offer a pro feature walkthrough.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            # Create mock message response
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            # Create async mock for messages.create
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # Set the class to return our mock
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert isinstance(result, RiskExplanation)
            assert result.reason == expected_response["reason"]
            assert result.action == expected_response["action"]

            # Verify API was called with correct model
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_args.kwargs["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_successful_explanation_without_shap_values(self, sample_user_context, sample_risk_inputs):
        """Test successful response when SHAP values are not available."""
        expected_response = {
            "reason": "Low engagement over the past month indicates reduced platform interest.",
            "action": "Send re-engagement email with new feature highlights.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-456"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=None,
                )

            assert isinstance(result, RiskExplanation)
            assert result.reason == expected_response["reason"]
            assert result.action == expected_response["action"]

    @pytest.mark.asyncio
    async def test_prompt_contains_user_context(self, sample_user_context, sample_risk_inputs):
        """Test that the prompt is correctly constructed with user context."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps({
                "reason": "Test reason.",
                "action": "Test action.",
            }))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            # Extract the prompt from the call
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt = messages[0]["content"]

            # Verify prompt contains key context
            assert "78/100" in prompt  # risk_score
            assert "pro" in prompt  # plan_type
            assert "180" in prompt  # days_since_signup
            assert "25" in prompt  # days_since_last_login
            assert "days_since_last_login" in prompt  # driver name


class TestExplainerMalformedJSON:
    """Test handling of malformed JSON responses."""

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, sample_user_context, sample_risk_inputs):
        """Test that malformed JSON falls back to default response."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="This is not JSON")]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert isinstance(result, RiskExplanation)
            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_json_missing_reason_field(self, sample_user_context, sample_risk_inputs):
        """Test that JSON missing reason field falls back to default."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps({"action": "Test action"}))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_json_missing_action_field(self, sample_user_context, sample_risk_inputs):
        """Test that JSON missing action field falls back to default."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps({"reason": "Test reason"}))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_json_with_empty_strings(self, sample_user_context, sample_risk_inputs):
        """Test that empty string values fall back to default."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps({"reason": "", "action": ""}))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."


class TestExplainerAPIErrors:
    """Test handling of API errors and timeouts."""

    @pytest.mark.asyncio
    async def test_api_timeout_error(self, sample_user_context, sample_risk_inputs):
        """Test that API timeout falls back to default response."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=anthropic.APIError(
                request=MagicMock(),
                message="Timeout",
                body={}
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_api_rate_limit_error(self, sample_user_context, sample_risk_inputs):
        """Test that rate limit errors fall back to default response."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=anthropic.APIError(
                request=MagicMock(),
                message="Rate limit",
                body={}
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_generic_api_error(self, sample_user_context, sample_risk_inputs):
        """Test that generic API errors fall back to default response."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=anthropic.APIError(
                request=MagicMock(),
                message="Server error",
                body={}
            ))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_unexpected_exception(self, sample_user_context, sample_risk_inputs):
        """Test that unexpected exceptions fall back to default response."""
        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=RuntimeError("Unexpected error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."


class TestExplainerMissingAPIKey:
    """Test handling of missing ANTHROPIC_API_KEY."""

    @pytest.mark.asyncio
    async def test_missing_api_key(self, sample_user_context, sample_risk_inputs):
        """Test that missing API key returns default explanation without calling Claude."""
        # Ensure API key is not set
        env_without_key = os.environ.copy()
        env_without_key.pop("ANTHROPIC_API_KEY", None)

        with patch.dict(os.environ, env_without_key, clear=True):
            result = await explain_risk(
                user_context=sample_user_context,
                risk_score=sample_risk_inputs["risk_score"],
                top_drivers=sample_risk_inputs["top_drivers"],
                shap_values=sample_risk_inputs["shap_values"],
            )

            assert result.reason == "Unable to analyze."
            assert result.action == "Contact support."

    @pytest.mark.asyncio
    async def test_api_key_from_environment(self, sample_user_context, sample_risk_inputs):
        """Test that API key is correctly read from environment."""
        expected_response = {
            "reason": "Test reason.",
            "action": "Test action.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            custom_key = "sk-test-custom-key-12345"
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": custom_key}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=sample_risk_inputs["risk_score"],
                    top_drivers=sample_risk_inputs["top_drivers"],
                    shap_values=sample_risk_inputs["shap_values"],
                )

                assert result.reason == expected_response["reason"]
                # Verify API was initialized with correct key
                mock_async_client_class.assert_called_once_with(api_key=custom_key, timeout=10.0)


class TestRiskExplanationDataclass:
    """Test RiskExplanation dataclass."""

    def test_risk_explanation_creation(self):
        """Test that RiskExplanation can be instantiated correctly."""
        reason = "User has not logged in for 30 days."
        action = "Schedule a retention call."

        explanation = RiskExplanation(reason=reason, action=action)

        assert explanation.reason == reason
        assert explanation.action == action

    def test_risk_explanation_immutability(self):
        """Test that RiskExplanation fields are accessible as dataclass attributes."""
        explanation = RiskExplanation(
            reason="Low engagement.",
            action="Send email.",
        )

        # Should be able to access fields
        assert hasattr(explanation, "reason")
        assert hasattr(explanation, "action")
        assert explanation.reason == "Low engagement."
        assert explanation.action == "Send email."


class TestExplainerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_top_drivers(self, sample_user_context):
        """Test with empty top_drivers list."""
        expected_response = {
            "reason": "User engagement is declining.",
            "action": "Reach out proactively.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=75,
                    top_drivers=[],
                    shap_values={},
                )

            assert isinstance(result, RiskExplanation)
            assert result.reason == expected_response["reason"]
            assert result.action == expected_response["action"]

    @pytest.mark.asyncio
    async def test_single_top_driver(self, sample_user_context):
        """Test with only one top driver."""
        expected_response = {
            "reason": "Primary engagement metric is weak.",
            "action": "Monitor closely.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=55,
                    top_drivers=["session_count_30d"],
                    shap_values={"session_count_30d": -0.25},
                )

            assert isinstance(result, RiskExplanation)
            assert result.reason == expected_response["reason"]

    @pytest.mark.asyncio
    async def test_risk_score_boundaries(self, sample_user_context):
        """Test with boundary risk scores (0 and 100)."""
        expected_response = {
            "reason": "High churn risk detected.",
            "action": "Urgent intervention needed.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                # Test with risk_score = 100
                result_high = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=100,
                    top_drivers=["days_since_last_login"],
                    shap_values={},
                )

                assert result_high.reason == expected_response["reason"]

                # Test with risk_score = 0
                result_low = await explain_risk(
                    user_context=sample_user_context,
                    risk_score=0,
                    top_drivers=["session_count_30d"],
                    shap_values={},
                )

                assert result_low.reason == expected_response["reason"]

    @pytest.mark.asyncio
    async def test_missing_user_context_fields(self):
        """Test with incomplete user context."""
        incomplete_context = {
            "plan_type": "free",
            # Missing days_since_signup and days_since_last_login
        }

        expected_response = {
            "reason": "Limited plan with low engagement.",
            "action": "Suggest plan upgrade.",
        }

        with patch("backend.app.ml.explainer.AsyncClient") as mock_async_client_class:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text=json.dumps(expected_response))]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client_class.return_value = mock_client

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                result = await explain_risk(
                    user_context=incomplete_context,
                    risk_score=65,
                    top_drivers=["plan_type_encoded"],
                    shap_values={},
                )

            assert isinstance(result, RiskExplanation)
            # Should use defaults (0) for missing fields
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt = messages[0]["content"]
            assert "0 days" in prompt  # default values for missing fields
