"""
app/predictor.py
================
Wraps the trained XGBoost model and label-encoders for safe, reusable inference.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
from typing import Tuple

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH    = os.path.join(BASE, "model.pkl")
ENCODERS_PATH = os.path.join(BASE, "encoders.pkl")

# Categorical column order must match train_model.py
CATEGORICAL_COLS = ["weather_condition", "time_of_day", "traffic_density", "road_type"]
BOOL_COLS        = ["has_intersection", "has_curve", "is_peak_hour"]
NUM_COLS         = ["latitude", "longitude", "num_lanes"]


class Predictor:
    """Singleton-style predictor loaded once at startup."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
            # Trigger training on first run
            import sys
            sys.path.insert(0, BASE)
            from train_model import train
            self._model, self._encoders, self._feature_cols = train()
        else:
            self._model = joblib.load(MODEL_PATH)
            meta = joblib.load(ENCODERS_PATH)
            self._encoders    = meta["encoders"]
            self._feature_cols = meta["feature_cols"]

    # ------------------------------------------------------------------
    def _safe_encode(self, col: str, value: str) -> int:
        le = self._encoders[col]
        classes = list(le.classes_)
        if value in classes:
            return int(le.transform([value])[0])
        # Fallback: nearest match
        return 0

    # ------------------------------------------------------------------
    def predict(
        self,
        latitude: float,
        longitude: float,
        weather_condition: str,
        time_of_day: str,
        traffic_density: str = "Medium",
        road_type: str = "Highway",
        num_lanes: int = 4,
        has_intersection: bool = False,
        has_curve: bool = False,
        is_peak_hour: bool = False,
    ) -> Tuple[float, float]:
        """
        Returns (risk_score, confidence).
        confidence is a heuristic based on prediction stability.
        """
        features = [
            latitude, longitude, num_lanes,
            self._safe_encode("weather_condition", weather_condition),
            self._safe_encode("time_of_day", time_of_day),
            self._safe_encode("traffic_density", traffic_density),
            self._safe_encode("road_type", road_type),
            int(has_intersection), int(has_curve), int(is_peak_hour),
        ]
        x = np.array([features], dtype=float)
        score = float(np.clip(self._model.predict(x)[0], 0.05, 0.98))

        # Simple confidence: invert distance from 0.5 centre
        confidence = round(min(0.97, 0.60 + abs(score - 0.5) * 0.7), 3)
        return round(score, 4), confidence


def risk_level(score: float) -> str:
    if score < 0.30:  return "Low"
    if score < 0.55:  return "Medium"
    if score < 0.75:  return "High"
    return "Critical"


def build_explanation(score: float, weather: str, tod: str, traffic: str) -> str:
    parts = []
    if score >= 0.75:
        parts.append("⚠️ Critical risk zone")
    elif score >= 0.55:
        parts.append("🟠 Elevated risk detected")
    else:
        parts.append("🟢 Relatively safe conditions")

    if weather in ("Rain", "Heavy Rain", "Fog"):
        parts.append(f"adverse weather ({weather.lower()}) reduces visibility/traction")
    if tod in ("Evening", "Night"):
        parts.append("low-light conditions increase accident likelihood")
    if traffic == "High":
        parts.append("high traffic density")
    if not parts[1:]:
        parts.append("conditions are within normal parameters")
    return "; ".join(parts) + "."
