"""Generate a fictional town ("Demoton") with a street network, LSOA-like zones, bike lanes
built over time, and census-style travel-to-work counts for 2011 and 2021.

The census counts are produced from a known effect of bike-lane exposure, so the model
in train_model.py has something real to recover. Swap this script for real data
(OS Open Roads / OSM, ONS LSOA boundaries, Census table TS061 / QS701) to make it real.
"""
import csv

import numpy as np

from config import (BACKGROUND_TREND, CENSUS_YEARS, NOISE_SD, RAW, SEED, STREET_SPACING_M,
                    TRUE_BETA_EXPOSURE, TRUE_BETA_EXPOSURE_SHORT, ZONE_SIZE_M, ZONES_PER_SIDE)
from geo import exposure_from_edges, write_geojson

rng = np.random.default_rng(SEED)
TOWN_M = ZONES_PER_SIDE * ZONE_SIZE_M
CENTRE = TOWN_M / 2


def make_zones():
    zones = []
    for row in range(ZONES_PER_SIDE):
        for col in range(ZONES_PER_SIDE):
            x0, y0 = col * ZONE_SIZE_M, row * ZONE_SIZE_M
            x1, y1 = x0 + ZONE_SIZE_M, y0 + ZONE_SIZE_M
            code = f"E01DEMO{row * ZONES_PER_SIDE + col + 1:03d}"
            dist_centre = np.hypot((x0 + x1) / 2 - CENTRE, (y0 + y1) / 2 - CENTRE)
            # Central zones have more short commutes (closer to jobs)
            short_share = float(np.clip(0.75 - dist_centre / TOWN_M + rng.normal(0, 0.05), 0.2, 0.85))
            zones.append({
                "type": "Feature",
                "properties": {
                    "lsoa_code": code,
                    "name": f"Demoton {row + 1}{'ABCDEF'[col]}",
                    "centroid": [(x0 + x1) / 2, (y0 + y1) / 2],
                    "short_commute_share": round(short_share, 3),
                },
                "geometry": {"type": "Polygon",
                             "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            })
    return zones


def lane_year(x1, y1, x2, y2):
    """Decide if/when a street segment got a bike lane. Returns a year or None."""
    horizontal = y1 == y2
    if horizontal and y1 == 1200 and x1 >= 800 and x2 <= 1600:
        return 2008  # an old town-centre lane, already there in 2011
    if not horizontal and x1 == 1000 and y1 >= 200:
        return 2015  # north-south corridor
    if horizontal and y1 == 1800 and x1 >= 400:
        return 2018  # east-west route in the north
    if not horizontal and x1 == 1800 and y2 <= 1200:
        return 2023  # built after the 2021 census: part of "today's" network only
    return None


def make_edges():
    edges, n = [], TOWN_M // STREET_SPACING_M
    for i in range(n + 1):
        for j in range(n):
            a = i * STREET_SPACING_M
            b0, b1 = j * STREET_SPACING_M, (j + 1) * STREET_SPACING_M
            for x1, y1, x2, y2 in ((b0, a, b1, a), (a, b0, a, b1)):  # horizontal, vertical
                edges.append({
                    "type": "Feature",
                    "properties": {"edge_id": f"e{len(edges):04d}",
                                   "lane_built_year": lane_year(x1, y1, x2, y2)},
                    "geometry": {"type": "LineString", "coordinates": [[x1, y1], [x2, y2]]},
                })
    return edges


def lanes_at(edges, year):
    return [e for e in edges if (e["properties"]["lane_built_year"] or 9999) <= year]


def make_census(zones, edges):
    rows = []
    y0, y1 = CENSUS_YEARS
    for z in zones:
        p = z["properties"]
        short = p["short_commute_share"]
        e0 = exposure_from_edges(p["centroid"], lanes_at(edges, y0))
        e1 = exposure_from_edges(p["centroid"], lanes_at(edges, y1))
        workers0 = int(rng.integers(550, 900))
        workers1 = int(workers0 * rng.uniform(0.95, 1.1))
        share0 = np.clip(0.015 + 0.03 * short + 0.004 * e0 + rng.normal(0, NOISE_SD), 0.005, 1)
        share1 = np.clip(share0 + BACKGROUND_TREND + TRUE_BETA_EXPOSURE * (e1 - e0)
                         + TRUE_BETA_EXPOSURE_SHORT * (e1 - e0) * short
                         + rng.normal(0, NOISE_SD), 0.005, 1)
        for year, workers, share in ((y0, workers0, share0), (y1, workers1, share1)):
            rows.append({"lsoa_code": p["lsoa_code"], "year": year, "workers": workers,
                         "cycle_to_work": int(round(share * workers))})
    return rows


def main():
    zones, edges = make_zones(), make_edges()
    write_geojson(RAW / "lsoas.geojson", zones)
    write_geojson(RAW / "edges.geojson", edges)
    census = make_census(zones, edges)
    with open(RAW / "census_travel_to_work.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=census[0].keys())
        w.writeheader()
        w.writerows(census)
    n_lanes = sum(e["properties"]["lane_built_year"] is not None for e in edges)
    print(f"Wrote {len(zones)} zones, {len(edges)} edges ({n_lanes} with lanes), "
          f"{len(census)} census rows to {RAW}")


if __name__ == "__main__":
    main()
