"""
app/route_engine.py
===================
Peddapalli road graph and safest-route computation.

Graph nodes = important junctions / landmarks in Peddapalli district.
Edges       = road segments connecting them, annotated with road metadata.

Algorithm:
  1. Find the 3 nearest graph nodes to origin and destination.
  2. Run Dijkstra three times with different cost functions:
       Route A – minimise total risk  (safest)
       Route B – minimise distance    (fastest)
       Route C – balanced (risk × distance)
  3. For each candidate path, call the ML predictor per segment.
  4. Rank routes by overall risk score and flag high-risk segments.
"""

from __future__ import annotations
import math
import heapq
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Road graph: nodes (id → lat/lng/label)
# ──────────────────────────────────────────────────────────────────────────────
NODES: Dict[str, Tuple[float, float, str]] = {
    # (lat, lng, label)
    "PDL": (18.6160, 79.3830, "Peddapalli Town"),
    "BSN": (18.7450, 79.4950, "Basanthnagar"),
    "GDK": (18.6600, 79.4900, "Godavarikhani"),
    "RMG": (18.7950, 79.4480, "Ramagundam"),
    "KRM": (18.5500, 79.3400, "Karimnagar Side"),
    "MTN": (18.5800, 79.5000, "Manthani"),
    "DHR": (18.6350, 79.3200, "Dharmaram"),
    "SLT": (18.5650, 79.4400, "Sulthanabad"),
    "MCH": (18.7500, 79.5100, "Mancherial Junction"),
    "ODE": (18.6400, 79.3050, "Odela"),
    "BYP": (18.6050, 79.3970, "Bypass Junction"),
    "NH1": (18.5300, 79.3600, "NH-163 South"),
    "IND": (18.7800, 79.4650, "Industrial Area"),
    "JKP": (18.6900, 79.4200, "Jaipuram"),
    "RMJ": (18.7200, 79.3800, "Ramakrishnapur"),
    "VYR": (18.6800, 79.3500, "Veeresalingam Nagar"),
    "ENK": (18.7000, 79.4600, "Enkoor"),
    "SRG": (18.6300, 79.4500, "Srirampur"),
}

# ──────────────────────────────────────────────────────────────────────────────
# Edges: (from, to, km, road_name, road_type, base_risk)
# ──────────────────────────────────────────────────────────────────────────────
_RAW_EDGES = [
    ("PDL", "GDK",  12.0, "Godavarikhani Road",           "Arterial", 0.38),
    ("PDL", "KRM",  14.0, "SH-1 Peddapalli-Karimnagar",   "Highway",  0.48),
    ("PDL", "DHR",   9.0, "Dharmaram Road",                "Local",    0.22),
    ("PDL", "BYP",   3.5, "Peddapalli Town Circle Road",   "Local",    0.18),
    ("PDL", "MTN",  22.0, "Manthani Road",                 "Arterial", 0.35),
    ("PDL", "SRG",   8.0, "Godavarikhani Road",            "Arterial", 0.36),
    ("BYP", "KRM",  11.0, "Karimnagar-Peddapalli Bypass",  "Highway",  0.46),
    ("BYP", "JKP",  10.0, "Rajiv Highway (SH-1)",          "Highway",  0.52),
    ("JKP", "GDK",   6.0, "Godavarikhani Road",            "Arterial", 0.40),
    ("JKP", "BSN",  10.0, "Rajiv Highway (SH-1)",          "Highway",  0.58),
    ("JKP", "RMJ",   8.0, "SH-7 Corridor",                 "Highway",  0.50),
    ("GDK", "ENK",   6.0, "Godavarikhani Road",            "Arterial", 0.42),
    ("GDK", "MCH",  12.0, "Mancherial Connector",          "Highway",  0.50),
    ("BSN", "RMG",  10.0, "Ramagundam Road",               "Highway",  0.60),
    ("BSN", "MCH",   8.0, "Basanthnagar Road",             "Arterial", 0.55),
    ("BSN", "ENK",   9.0, "Basanthnagar Road",             "Arterial", 0.52),
    ("RMG", "IND",   5.0, "Industrial Bypass Ramagundam",  "Arterial", 0.44),
    ("RMG", "MCH",  10.0, "Ramagundam Road",               "Highway",  0.58),
    ("MCH", "IND",   6.0, "Mancherial Connector",          "Highway",  0.50),
    ("ODE", "DHR",   6.0, "Odela-Peddapalli Road",         "Arterial", 0.30),
    ("ODE", "PDL",  12.0, "Odela-Peddapalli Road",         "Arterial", 0.36),
    ("KRM", "NH1",  10.0, "NH-163 Stretch",                "Highway",  0.48),
    ("NH1", "SLT",   8.0, "NH-163 Stretch",                "Highway",  0.45),
    ("SLT", "MTN",  12.0, "Manthani Road",                 "Arterial", 0.32),
    ("RMJ", "VYR",   7.0, "SH-7 Corridor",                 "Highway",  0.50),
    ("VYR", "DHR",   6.0, "Dharmaram Road",                "Local",    0.24),
    ("ENK", "IND",   8.0, "Industrial Bypass Ramagundam",  "Arterial", 0.42),
    ("SRG", "GDK",   5.0, "Godavarikhani Road",            "Arterial", 0.38),
    ("SRG", "MTN",  14.0, "Manthani Road",                 "Arterial", 0.34),
    ("RMJ", "ENK",   9.0, "SH-7 Corridor",                 "Highway",  0.52),
]

# Build adjacency list (bidirectional)
GRAPH: Dict[str, List[Tuple[str, float, str, str, float]]] = {n: [] for n in NODES}
for u, v, km, rname, rtype, brisk in _RAW_EDGES:
    GRAPH[u].append((v, km, rname, rtype, brisk))
    GRAPH[v].append((u, km, rname, rtype, brisk))


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def _haversine(lat1, lng1, lat2, lng2) -> float:
    """Distance in km between two lat/lng points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_nodes(lat: float, lng: float, k: int = 2) -> List[str]:
    """Return the k nearest graph nodes to a coordinate."""
    dists = [(n, _haversine(lat, lng, nlat, nlng))
             for n, (nlat, nlng, _) in NODES.items()]
    dists.sort(key=lambda x: x[1])
    return [n for n, _ in dists[:k]]


# ──────────────────────────────────────────────────────────────────────────────
# Dijkstra
# ──────────────────────────────────────────────────────────────────────────────
def _dijkstra(start: str, end: str, weight_fn) -> Optional[List[str]]:
    """Generic Dijkstra; weight_fn(u, v, km, base_risk) → cost."""
    dist = {n: float("inf") for n in NODES}
    dist[start] = 0.0
    prev: Dict[str, Optional[str]] = {n: None for n in NODES}
    pq = [(0.0, start)]

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist[u]:
            continue
        if u == end:
            break
        for v, km, _, _, brisk in GRAPH[u]:
            w = weight_fn(u, v, km, brisk)
            nc = dist[u] + w
            if nc < dist[v]:
                dist[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))

    if dist[end] == float("inf"):
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def compute_routes(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    predictor,
    time_of_day: str = "Morning",
    weather: str = "Clear",
    traffic: str = "Medium",
) -> List[dict]:
    """
    Returns up to 3 route candidates, each with full segment details and
    an overall risk score.  The list is sorted best (safest) first.
    """
    origins = nearest_nodes(origin_lat, origin_lng, k=1)
    dests   = nearest_nodes(dest_lat, dest_lng, k=1)

    start = origins[0]
    end   = dests[0]

    # Three cost functions → three diverse routes
    weight_fns = [
        # (label, fn)
        ("Safest",   lambda u, v, km, br: br * 10 + km * 0.05),        # minimise risk
        ("Fastest",  lambda u, v, km, br: km + br * 1.0),              # minimise distance
        ("Balanced", lambda u, v, km, br: km * 0.5 + br * 5 + 0.3 * km * br),
    ]

    results = []
    seen_paths = set()

    for label, wfn in weight_fns:
        path = _dijkstra(start, end, wfn)
        if path is None:
            continue
        key = "→".join(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        segments = []
        total_risk = 0.0
        total_km = 0.0
        high_risk_count = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            u_lat, u_lng, _ = NODES[u]
            v_lat, v_lng, _ = NODES[v]

            # Find edge metadata
            edge_meta = next(
                (e for e in GRAPH[u] if e[0] == v), None
            )
            if edge_meta is None:
                continue
            _, km, rname, rtype, base_risk = edge_meta

            # ML prediction for this segment
            score, _ = predictor.predict(
                latitude          = (u_lat + v_lat) / 2,
                longitude         = (u_lng + v_lng) / 2,
                weather_condition = weather,
                time_of_day       = time_of_day,
                traffic_density   = traffic,
                road_type         = rtype,
                num_lanes         = 4 if rtype == "Highway" else 2,
                has_intersection  = False,
                has_curve         = rtype == "Highway",
                is_peak_hour      = time_of_day in ("Morning", "Evening"),
            )

            rl = _risk_level(score)
            warning = None
            if score > 0.70:
                warning = f"High-risk stretch on {rname} – consider alternate roads"
                high_risk_count += 1
            elif score > 0.50:
                warning = f"Moderate caution advised on {rname}"

            segments.append({
                "from_lat":   round(u_lat, 6),
                "from_lng":   round(u_lng, 6),
                "to_lat":     round(v_lat, 6),
                "to_lng":     round(v_lng, 6),
                "road_name":  rname,
                "risk_score": round(score, 4),
                "risk_level": rl,
                "warning":    warning,
                "km":         round(km, 1),
            })
            total_risk += score
            total_km   += km

        if not segments:
            continue

        overall = round(total_risk / len(segments), 4)
        results.append({
            "label":           label,
            "path_nodes":      path,
            "segments":        segments,
            "overall_risk":    overall,
            "estimated_km":    round(total_km, 1),
            "high_risk_count": high_risk_count,
        })

    # Sort by overall_risk ascending (safest first)
    results.sort(key=lambda r: (r["high_risk_count"], r["overall_risk"]))
    return results


def _risk_level(score: float) -> str:
    if score < 0.30:  return "Low"
    if score < 0.55:  return "Medium"
    if score < 0.75:  return "High"
    return "Critical"


def node_label(node_id: str) -> str:
    return NODES.get(node_id, (0, 0, node_id))[2]
