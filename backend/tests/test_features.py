"""
Unit tests for feature engineering pipeline (backend/app/ml/features.py).

Tests:
- extract_user_features returns correct dict shape (5 required keys)
- extract_user_features handles empty events list (raises ValueError)
- extract_user_features returns zero values for users with no activity
- Feature values are floats and have no NaN
- Songs played total is counted correctly
- Thumbs up/down sentiment counting works
- Playlist feature counting works
- Average session duration calculation
"""

import pytest
from datetime import datetime, timedelta
from backend.app.ml.features import extract_user_features


class MockEvent:
    """Mock event object for testing feature extraction."""

    def __init__(self, event_type, occurred_at, event_metadata=None):
        self.event_type = event_type
        self.occurred_at = occurred_at
        self.event_metadata = event_metadata


class TestExtractUserFeatures:
    """Test extract_user_features function."""

    def test_extract_returns_correct_dict_shape(self):
        """Test that extract_user_features returns dict with all 5 required keys."""
        events = [MockEvent("feature_used", datetime.utcnow(), None)]
        result = extract_user_features(events)

        assert isinstance(result, dict)
        required_keys = {
            "songs_played_total",
            "thumbs_down_count",
            "thumbs_up_count",
            "add_to_playlist_count",
            "avg_session_duration_min",
        }
        assert set(result.keys()) == required_keys

    def test_extract_empty_events_raises_value_error(self):
        """Test that empty events list raises ValueError."""
        with pytest.raises(ValueError, match="User events list is empty"):
            extract_user_features([])

    def test_extract_returns_float_values(self):
        """Test that all returned values are floats."""
        events = [MockEvent("feature_used", datetime.utcnow(), None)]
        result = extract_user_features(events)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_extract_no_nan_values(self):
        """Test that result contains no NaN values."""
        events = [MockEvent("feature_used", datetime.utcnow(), None)]
        result = extract_user_features(events)

        for key, value in result.items():
            assert not (isinstance(value, float) and value != value), f"{key} is NaN"  # NaN != NaN

    def test_songs_played_total_counts_feature_used_events(self):
        """Test that songs_played_total counts feature_used events."""
        now = datetime.utcnow()
        events = [
            MockEvent("feature_used", now),
            MockEvent("feature_used", now),
            MockEvent("login", now),
            MockEvent("support_ticket", now),
        ]

        result = extract_user_features(events)

        assert result["songs_played_total"] == 2.0

    def test_thumbs_up_counts_positive_sentiment(self):
        """Test that thumbs_up_count counts support_ticket with positive sentiment."""
        now = datetime.utcnow()
        events = [
            MockEvent("support_ticket", now, {"sentiment": "positive"}),
            MockEvent("support_ticket", now, {"sentiment": "positive"}),
            MockEvent("support_ticket", now, {"sentiment": "negative"}),
        ]

        result = extract_user_features(events)

        assert result["thumbs_up_count"] == 2.0

    def test_thumbs_down_counts_negative_sentiment(self):
        """Test that thumbs_down_count counts support_ticket with negative sentiment."""
        now = datetime.utcnow()
        events = [
            MockEvent("support_ticket", now, {"sentiment": "negative"}),
            MockEvent("support_ticket", now, {"sentiment": "negative"}),
            MockEvent("support_ticket", now, {"sentiment": "positive"}),
        ]

        result = extract_user_features(events)

        assert result["thumbs_down_count"] == 2.0

    def test_add_to_playlist_counts_playlist_features(self):
        """Test that add_to_playlist_count counts feature_used with playlist feature_name."""
        now = datetime.utcnow()
        events = [
            MockEvent("feature_used", now, {"feature_name": "playlist"}),
            MockEvent("feature_used", now, {"feature_name": "playlist"}),
            MockEvent("feature_used", now, {"feature_name": "search"}),
        ]

        result = extract_user_features(events)

        assert result["add_to_playlist_count"] == 2.0

    def test_avg_session_duration_with_time_span(self):
        """Test that avg_session_duration_min is calculated correctly."""
        start = datetime.utcnow()
        end = start + timedelta(days=10)

        events = [
            MockEvent("login", start),
            MockEvent("feature_used", end),
        ]

        result = extract_user_features(events)

        # 10 days * 24 hours * 60 min / 2 events = 7200 min per event
        expected = (10 * 24 * 60) / 2
        assert result["avg_session_duration_min"] == expected

    def test_avg_session_duration_single_event_is_zero(self):
        """Test that avg_session_duration_min is 0 when all events are on same day."""
        now = datetime.utcnow()

        events = [
            MockEvent("login", now),
            MockEvent("feature_used", now),
        ]

        result = extract_user_features(events)

        # Same day -> 0 days -> 0 duration
        assert result["avg_session_duration_min"] == 0.0

    def test_extract_with_missing_metadata_fields(self):
        """Test that extract handles missing metadata gracefully."""
        now = datetime.utcnow()
        events = [
            MockEvent("support_ticket", now, None),  # No metadata
            MockEvent("feature_used", now, {}),  # Empty metadata
        ]

        result = extract_user_features(events)

        # No positive/negative sentiment found
        assert result["thumbs_up_count"] == 0.0
        assert result["thumbs_down_count"] == 0.0
        assert result["add_to_playlist_count"] == 0.0

    def test_extract_sorts_events_by_time(self):
        """Test that events are sorted by occurred_at before processing."""
        now = datetime.utcnow()
        events = [
            MockEvent("login", now + timedelta(days=5)),
            MockEvent("login", now),
            MockEvent("login", now + timedelta(days=10)),
        ]

        result = extract_user_features(events)

        # Time span should be 10 days regardless of input order
        expected = (10 * 24 * 60) / 3
        assert result["avg_session_duration_min"] == expected

    def test_extract_with_observation_date_parameter(self):
        """Test that observation_date parameter is accepted (for future use)."""
        now = datetime.utcnow()
        past_date = now - timedelta(days=30)

        events = [
            MockEvent("login", now),
        ]

        # Should not raise error with observation_date
        result = extract_user_features(events, observation_date=past_date)

        assert isinstance(result, dict)

    def test_extract_with_all_event_types(self):
        """Test that extract handles all event types correctly."""
        now = datetime.utcnow()
        events = [
            MockEvent("login", now),
            MockEvent("feature_used", now, {"feature_name": "search"}),
            MockEvent("support_ticket", now, {"sentiment": "positive"}),
        ]

        result = extract_user_features(events)

        # All values should be computed without error
        assert result["songs_played_total"] == 1.0
        assert result["thumbs_up_count"] == 1.0

    def test_extract_large_number_of_events(self):
        """Test that extract handles large event lists efficiently."""
        now = datetime.utcnow()
        events = [
            MockEvent("feature_used", now + timedelta(hours=i))
            for i in range(1000)
        ]

        result = extract_user_features(events)

        assert result["songs_played_total"] == 1000.0
        assert isinstance(result, dict)

    def test_extract_with_no_matching_features(self):
        """Test that extract returns zeros for unused features."""
        now = datetime.utcnow()
        events = [
            MockEvent("login", now),
        ]

        result = extract_user_features(events)

        # Login events don't match any feature
        assert result["songs_played_total"] == 0.0
        assert result["thumbs_up_count"] == 0.0
        assert result["thumbs_down_count"] == 0.0
        assert result["add_to_playlist_count"] == 0.0

    def test_extract_feature_used_with_playlist_metadata(self):
        """Test playlist feature detection with various metadata."""
        now = datetime.utcnow()
        events = [
            MockEvent("feature_used", now, {"feature_name": "playlist", "action": "add"}),
        ]

        result = extract_user_features(events)

        # Should count as both songs_played and add_to_playlist
        assert result["songs_played_total"] == 1.0
        assert result["add_to_playlist_count"] == 1.0

    def test_extract_support_ticket_without_sentiment(self):
        """Test support ticket without sentiment metadata."""
        now = datetime.utcnow()
        events = [
            MockEvent("support_ticket", now, {"issue": "bug"}),  # No sentiment
        ]

        result = extract_user_features(events)

        # Should not count without sentiment
        assert result["thumbs_up_count"] == 0.0
        assert result["thumbs_down_count"] == 0.0
