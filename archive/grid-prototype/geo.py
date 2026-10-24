"""Tiny geometry helpers (no GIS dependencies, on purpose)."""
import json
import math

from config import DECAY_M


def edge_length(coords):
    (x1, y1), (x2, y2) = coords
    return math.hypot(x2 - x1, y2 - y1)


def edge_midpoint(coords):
    (x1, y1), (x2, y2) = coords
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def exposure_from_edges(centroid, lane_edges):
    """Bike-lane exposure for one zone: sum of lane km, distance-decayed from the zone centroid.

    exposure = sum_e  length_km(e) * exp(-distance(centroid, midpoint(e)) / DECAY_M)
    The same formula is implemented in frontend/app.js.
    """
    cx, cy = centroid
    total = 0.0
    for e in lane_edges:
        coords = e["geometry"]["coordinates"]
        mx, my = edge_midpoint(coords)
        d = math.hypot(mx - cx, my - cy)
        total += edge_length(coords) / 1000 * math.exp(-d / DECAY_M)
    return total


def read_geojson(path):
    with open(path) as f:
        return json.load(f)


def write_geojson(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
