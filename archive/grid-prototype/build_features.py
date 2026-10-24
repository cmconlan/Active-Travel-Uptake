"""Derive LSOA-level features from network changes between census years.

Output: data/processed/features.csv with one row per LSOA:
  exposure_2011, exposure_2021, d_exposure, short_commute_share,
  cycle_share_2011, cycle_share_2021, d_cycle_share
"""
import csv
from collections import defaultdict

from config import CENSUS_YEARS, PROCESSED, RAW
from geo import exposure_from_edges, read_geojson


def lanes_at(edges, year):
    return [e for e in edges if (e["properties"]["lane_built_year"] or 9999) <= year]


def main():
    zones = read_geojson(RAW / "lsoas.geojson")["features"]
    edges = read_geojson(RAW / "edges.geojson")["features"]

    census = defaultdict(dict)
    with open(RAW / "census_travel_to_work.csv") as f:
        for r in csv.DictReader(f):
            census[r["lsoa_code"]][int(r["year"])] = int(r["cycle_to_work"]) / int(r["workers"])

    y0, y1 = CENSUS_YEARS
    lanes0, lanes1 = lanes_at(edges, y0), lanes_at(edges, y1)
    rows = []
    for z in zones:
        p = z["properties"]
        e0 = exposure_from_edges(p["centroid"], lanes0)
        e1 = exposure_from_edges(p["centroid"], lanes1)
        s0, s1 = census[p["lsoa_code"]][y0], census[p["lsoa_code"]][y1]
        rows.append({
            "lsoa_code": p["lsoa_code"],
            f"exposure_{y0}": round(e0, 4), f"exposure_{y1}": round(e1, 4),
            "d_exposure": round(e1 - e0, 4),
            "short_commute_share": p["short_commute_share"],
            f"cycle_share_{y0}": round(s0, 4), f"cycle_share_{y1}": round(s1, 4),
            "d_cycle_share": round(s1 - s0, 4),
        })

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote features for {len(rows)} LSOAs to {PROCESSED / 'features.csv'}")


if __name__ == "__main__":
    main()
