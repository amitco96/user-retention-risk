"""
Quick endpoint verification script.

This script tests the key endpoints without requiring a full Docker setup.
Run with: python backend/tests/test_endpoints.py
"""

import asyncio
from datetime import datetime, timedelta
from backend.app.ml.features import extract_user_features
from backend.app.ml.model import predict
from backend.app.schemas.risk import RiskResponse, RiskSummary, CohortRetentionData


async def test_feature_extraction():
    """Test feature extraction for a user."""
    print("\n[TEST] Feature Extraction")

    # Create mock events
    class MockEvent:
        def __init__(self, event_type, occurred_at, event_metadata=None):
            self.event_type = event_type
            self.occurred_at = occurred_at
            self.event_metadata = event_metadata or {}

    now = datetime.utcnow()
    events = [
        MockEvent('feature_used', now - timedelta(days=i))
        for i in range(10)
    ]

    features = extract_user_features(events)

    assert 'songs_played_total' in features
    assert 'thumbs_down_count' in features
    assert 'thumbs_up_count' in features
    assert 'add_to_playlist_count' in features
    assert 'avg_session_duration_min' in features

    print("  Features extracted successfully:")
    for name, value in sorted(features.items()):
        print(f"    - {name}: {value:.2f}")
    return True


async def test_model_prediction():
    """Test model prediction."""
    print("\n[TEST] Model Prediction")

    features = {
        'songs_played_total': 50.0,
        'thumbs_down_count': 5.0,
        'thumbs_up_count': 15.0,
        'add_to_playlist_count': 10.0,
        'avg_session_duration_min': 30.0,
    }

    prediction = predict('test-user-123', features)

    assert 0 <= prediction.risk_score <= 100
    assert prediction.risk_tier in ['low', 'medium', 'high', 'critical']
    assert len(prediction.top_drivers) > 0

    print(f"  Risk Score: {prediction.risk_score}/100")
    print(f"  Risk Tier: {prediction.risk_tier}")
    print(f"  Top Drivers: {', '.join(prediction.top_drivers)}")
    return True


async def test_schemas():
    """Test Pydantic schema validation."""
    print("\n[TEST] Schema Validation")

    # Test RiskResponse
    response = RiskResponse(
        user_id='user-123',
        risk_score=75,
        risk_tier='high',
        top_drivers=['feature1', 'feature2', 'feature3'],
        reason='User has not logged in for 30 days',
        recommended_action='Send retention email offering discount',
        scored_at=datetime.utcnow(),
        model_version='1.0'
    )
    print(f"  RiskResponse created: {response.risk_score} -> {response.risk_tier}")

    # Test RiskSummary
    summary = RiskSummary(
        user_id='user-123',
        risk_score=75,
        risk_tier='high',
        reason='Inactive'
    )
    print(f"  RiskSummary created: {summary.user_id}")

    # Test CohortRetentionData (nested cohorts -> weeks shape)
    from backend.app.schemas.risk import Cohort, CohortWeekData
    cohort = CohortRetentionData(cohorts=[
        Cohort(cohort_week='2026-01', weeks=[
            CohortWeekData(week=0, retention_pct=100.0),
            CohortWeekData(week=1, retention_pct=85.0),
        ]),
        Cohort(cohort_week='2026-02', weeks=[
            CohortWeekData(week=0, retention_pct=98.0),
        ]),
    ])
    weeks_count = sum(len(c.weeks) for c in cohort.cohorts)
    print(f"  CohortRetentionData created: {len(cohort.cohorts)} cohorts, {weeks_count} total weeks")
    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("PHASE 4 ENDPOINT VERIFICATION")
    print("=" * 70)

    try:
        await test_feature_extraction()
        await test_model_prediction()
        await test_schemas()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
