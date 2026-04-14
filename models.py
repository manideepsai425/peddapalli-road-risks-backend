"""
app/models.py
=============
Pydantic v2 request / response models for the Road Risk API.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums / literals
# ---------------------------------------------------------------------------

WEATHER_OPTIONS = ["Clear", "Rain", "Fog", "Heavy Rain"]
TIME_OPTIONS    = ["Morning", "Afternoon", "Evening", "Night"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]
ROAD_TYPES      = ["Highway", "Arterial", "Local"]


# ---------------------------------------------------------------------------
# /predict/risk
# ---------------------------------------------------------------------------

class RiskRequest(BaseModel):
    latitude:          float  = Field(..., ge=18.40, le=18.90, example=18.616)
    longitude:         float  = Field(..., ge=79.10, le=79.70, example=79.383)
    weather_condition: str    = Field(..., example="Rain")
    time_of_day:       str    = Field(..., example="Night")
    traffic_density:   str    = Field("Medium", example="High")
    road_type:         str    = Field("Highway", example="Highway")
    num_lanes:         int    = Field(4, ge=2, le=6, example=4)
    has_intersection:  bool   = Field(False)
    has_curve:         bool   = Field(False)
    is_peak_hour:      bool   = Field(False)


class RiskResponse(BaseModel):
    risk_score:  float
    risk_level:  str          # "Low" | "Medium" | "High" | "Critical"
    confidence:  float
    explanation: str


# ---------------------------------------------------------------------------
# /predict/route
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    origin_lat:      float = Field(..., ge=18.40, le=18.90, example=18.616)
    origin_lng:      float = Field(..., ge=79.10, le=79.70, example=79.383)
    dest_lat:        float = Field(..., ge=18.40, le=18.90, example=18.750)
    dest_lng:        float = Field(..., ge=79.10, le=79.70, example=79.500)
    preferred_time:  str   = Field("Morning", example="Evening")
    weather:         str   = Field("Clear", example="Rain")
    traffic_density: str   = Field("Medium", example="High")


class RouteSegment(BaseModel):
    from_lat:     float
    from_lng:     float
    to_lat:       float
    to_lng:       float
    road_name:    str
    risk_score:   float
    risk_level:   str
    warning:      Optional[str] = None


class RouteAlternative(BaseModel):
    route_id:         str        # "Route_1", "Route_2" …
    label:            str        # "Safest", "Faster", "Scenic"
    segments:         List[RouteSegment]
    overall_risk:     float
    estimated_km:     float
    high_risk_count:  int
    risk_breakdown:   str        # human-readable summary
    is_recommended:   bool


class RouteResponse(BaseModel):
    safest_route:        RouteAlternative
    alternatives:        List[RouteAlternative]
    origin_label:        str
    destination_label:   str
    analysis_summary:    str


# ---------------------------------------------------------------------------
# /hotspots
# ---------------------------------------------------------------------------

class HotspotPoint(BaseModel):
    latitude:     float
    longitude:    float
    risk_score:   float
    road_name:    str
    incident_count: int


# ---------------------------------------------------------------------------
# /analytics
# ---------------------------------------------------------------------------

class AnalyticsSummary(BaseModel):
    total_accidents:        int
    high_risk_segments:     int
    most_dangerous_road:    str
    peak_accident_time:     str
    peak_weather:           str
    avg_risk_score:         float
    by_severity:            dict
    by_time_of_day:         dict
    by_weather:             dict
    by_road_type:           dict
    monthly_trend:          List[dict]
