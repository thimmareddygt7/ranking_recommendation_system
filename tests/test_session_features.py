from datetime import datetime, timedelta
from src.features.session_features import SessionFeatureExtractor

def test_empty_session():
    extractor = SessionFeatureExtractor(session_window_minutes=15)
    features = extractor.extract_realtime_signals([])
    assert features["session_event_count"] == 0.0
    assert features["session_dwell_seconds"] == 0.0

def test_session_window_filtering():
    extractor = SessionFeatureExtractor(session_window_minutes=15)
    now = datetime.utcnow()

    events = [
        {"timestamp": now - timedelta(minutes=25), "item_id": 101, "event": "view"}, # Outside window
        {"timestamp": now - timedelta(minutes=10), "item_id": 102, "event": "view"}, # Inside window
        {"timestamp": now - timedelta(minutes=2), "item_id": 103, "event": "addtocart"}, # Inside window
    ]

    signals = extractor.extract_realtime_signals(events)
    assert signals["session_event_count"] == 2.0
    assert signals["session_cart_count"] == 1.0
    assert signals["session_dwell_seconds"] > 0.0
