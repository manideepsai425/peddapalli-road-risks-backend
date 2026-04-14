"""
app/main.py
===========
FastAPI application for Peddapalli Road Accident Prediction.

Endpoints
---------
GET  /health
POST /predict/risk   – single-point risk prediction
POST /predict/route  – safest route between two points
GET  /hotspots       – clustered accident hotspots
GET  /analytics      – dataset-level statistics
GET  /segments       – raw GeoJSON road segments
"""

from __future__ import annotations
import json
import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    RiskRequest, RiskResponse,
    RouteRequest, RouteResponse, RouteAlternative, RouteSegment,
    HotspotPoint, AnalyticsSummary,
)
from app.predictor import Predictor, risk_level, build_explanation
from app.route_engine import compute_routes, node_label, NODES, nearest_nodes

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "accidents_data.csv")
GEO_PATH  = os.path.join(BASE_DIR, "data", "road_segments.geojson")

app = FastAPI(
    title="Peddapalli Road Risk AI",
    description="AI-powered accident risk prediction for Peddapalli district, Telangana.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data at startup
_predictor: Predictor | None = None
_df: pd.DataFrame | None = None


@app.on_event("startup")
async def startup():
    global _predictor, _df
    _predictor = Predictor()
    _df = pd.read_csv(DATA_PATH)
    print("✅ Predictor and dataset ready.")


def get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(500, "Model not loaded yet.")
    return _predictor


def get_df() -> pd.DataFrame:
    if _df is None:
        raise HTTPException(500, "Dataset not loaded yet.")
    return _df


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "rows_loaded":  len(_df) if _df is not None else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict/risk
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict/risk", response_model=RiskResponse, tags=["Prediction"])
def predict_risk(req: RiskRequest):
    p = get_predictor()
    score, confidence = p.predict(
        latitude          = req.latitude,
        longitude         = req.longitude,
        weather_condition = req.weather_condition,
        time_of_day       = req.time_of_day,
        traffic_density   = req.traffic_density,
        road_type         = req.road_type,
        num_lanes         = req.num_lanes,
        has_intersection  = req.has_intersection,
        has_curve         = req.has_curve,
        is_peak_hour      = req.is_peak_hour,
    )
    return RiskResponse(
        risk_score  = score,
        risk_level  = risk_level(score),
        confidence  = confidence,
        explanation = build_explanation(score, req.weather_condition,
                                        req.time_of_day, req.traffic_density),
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /predict/route
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/predict/route", response_model=RouteResponse, tags=["Prediction"])
def predict_route(req: RouteRequest):
    p = get_predictor()

    raw_routes = compute_routes(
        origin_lat  = req.origin_lat,
        origin_lng  = req.origin_lng,
        dest_lat    = req.dest_lat,
        dest_lng    = req.dest_lng,
        predictor   = p,
        time_of_day = req.preferred_time,
        weather     = req.weather,
        traffic     = req.traffic_density,
    )

    if not raw_routes:
        raise HTTPException(404, "No route found between specified coordinates.")

    origin_node = nearest_nodes(req.origin_lat, req.origin_lng, k=1)[0]
    dest_node   = nearest_nodes(req.dest_lat,   req.dest_lng,   k=1)[0]

    def _build_alt(r: dict, idx: int, is_rec: bool) -> RouteAlternative:
        segs = [RouteSegment(**{k: v for k, v in s.items() if k != "km"})
                for s in r["segments"]]
        ov   = r["overall_risk"]
        hrc  = r["high_risk_count"]
        breakdown = _risk_breakdown(ov, hrc, r["estimated_km"], r["label"])
        return RouteAlternative(
            route_id         = f"Route_{idx+1}",
            label            = r["label"],
            segments         = segs,
            overall_risk     = ov,
            estimated_km     = r["estimated_km"],
            high_risk_count  = hrc,
            risk_breakdown   = breakdown,
            is_recommended   = is_rec,
        )

    alts = [_build_alt(r, i, i == 0) for i, r in enumerate(raw_routes)]
    safest = alts[0]
    others = alts[1:]

    summary = _route_summary(safest, req.weather, req.preferred_time)

    return RouteResponse(
        safest_route      = safest,
        alternatives      = others,
        origin_label      = node_label(origin_node),
        destination_label = node_label(dest_node),
        analysis_summary  = summary,
    )


def _risk_breakdown(ov: float, hrc: int, km: float, label: str) -> str:
    level = "🟢 Low" if ov < 0.35 else ("🟠 Moderate" if ov < 0.60 else "🔴 High")
    return (
        f"{label} route | {km:.1f} km | Overall risk: {ov:.2f} ({level}) | "
        f"High-risk segments: {hrc}"
    )


def _route_summary(route: RouteAlternative, weather: str, tod: str) -> str:
    warnings = [s.warning for s in route.segments if s.warning]
    parts = [
        f"Recommended safest route spans {route.estimated_km:.1f} km "
        f"with an overall risk score of {route.overall_risk:.2f}."
    ]
    if weather in ("Rain", "Heavy Rain", "Fog"):
        parts.append(f"Adverse weather ({weather}) increases risk — drive carefully.")
    if tod in ("Evening", "Night"):
        parts.append("Night-time travel adds risk; ensure headlights are working.")
    if warnings:
        parts.append("Unavoidable caution zones: " + " | ".join(set(warnings[:3])))
    else:
        parts.append("No critical high-risk segments on this route.")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# GET /hotspots
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/hotspots", response_model=List[HotspotPoint], tags=["Analytics"])
def hotspots(min_risk: float = 0.55, limit: int = 60):
    df = get_df()
    high = df[df["risk_score"] >= min_risk].copy()

    # Cluster by rounding coordinates to ~0.5 km grid
    high["lat_bin"] = (high["latitude"]  / 0.005).round() * 0.005
    high["lng_bin"] = (high["longitude"] / 0.005).round() * 0.005

    grouped = (
        high.groupby(["lat_bin", "lng_bin"])
        .agg(
            risk_score   = ("risk_score",  "mean"),
            road_name    = ("road_name",   lambda x: x.mode()[0]),
            incident_count = ("accident_id", "count"),
        )
        .reset_index()
        .sort_values("risk_score", ascending=False)
        .head(limit)
    )

    return [
        HotspotPoint(
            latitude      = round(row.lat_bin, 6),
            longitude     = round(row.lng_bin, 6),
            risk_score    = round(row.risk_score, 4),
            road_name     = row.road_name,
            incident_count = int(row.incident_count),
        )
        for row in grouped.itertuples()
    ]


# ──────────────────────────────────────────────────────────────────────────────
# GET /analytics
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/analytics", response_model=AnalyticsSummary, tags=["Analytics"])
def analytics():
    df = get_df()

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["month"]     = df["date_time"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month")
        .agg(count=("accident_id", "count"), avg_risk=("risk_score", "mean"))
        .reset_index()
        .sort_values("month")
        .tail(18)
    )

    return AnalyticsSummary(
        total_accidents     = len(df),
        high_risk_segments  = int((df["risk_score"] >= 0.70).sum()),
        most_dangerous_road = df.groupby("road_name")["risk_score"].mean().idxmax(),
        peak_accident_time  = df["time_of_day"].value_counts().idxmax(),
        peak_weather        = df["weather_condition"].value_counts().idxmax(),
        avg_risk_score      = round(float(df["risk_score"].mean()), 4),
        by_severity         = df["accident_severity"].value_counts().to_dict(),
        by_time_of_day      = df["time_of_day"].value_counts().to_dict(),
        by_weather          = df["weather_condition"].value_counts().to_dict(),
        by_road_type        = df["road_type"].value_counts().to_dict(),
        monthly_trend       = monthly.to_dict(orient="records"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /segments  (raw GeoJSON)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/segments", tags=["Data"])
def segments():
    with open(GEO_PATH) as f:
        return json.load(f)
