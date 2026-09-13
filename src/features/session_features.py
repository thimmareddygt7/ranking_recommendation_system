import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class SessionFeatureExtractor:
    """
    Extracts short-term, real-time session signals from recent user interactions.
    """
    def __init__(self, session_window_minutes: int = 15):
        self.session_window = timedelta(minutes=session_window_minutes)

    def extract_realtime_signals(self, recent_events: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates session-level momentum metrics from a list of recent event dicts:
        [{'timestamp': datetime, 'item_id': int, 'event': str}, ...]
        """
        if not recent_events:
            return {
                "session_event_count": 0.0,
                "session_cart_count": 0.0,
                "session_view_cart_ratio": 0.0,
                "session_dwell_seconds": 0.0,
            }

        df = pd.DataFrame(recent_events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        latest_time = df["timestamp"].max()
        window_start = latest_time - self.session_window
        in_session_df = df[df["timestamp"] >= window_start]

        total_events = len(in_session_df)
        cart_events = (in_session_df["event"] == "addtocart").sum()
        view_events = (in_session_df["event"] == "view").sum()

        dwell_seconds = 0.0
        if total_events > 1:
            dwell_seconds = (in_session_df["timestamp"].max() - in_session_df["timestamp"].min()).total_seconds()

        return {
            "session_event_count": float(total_events),
            "session_cart_count": float(cart_events),
            "session_view_cart_ratio": float(cart_events / (view_events + 1e-5)),
            "session_dwell_seconds": float(dwell_seconds),
        }
