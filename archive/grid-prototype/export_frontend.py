"""Write the small, static data files the front end needs (no server required)."""
import csv
import datetime as dt

from config import CENSUS_YEARS, FRONTEND_DATA, RAW
from geo import read_geojson, write_geojson


def main():
    today = dt.date.today().year
    latest = CENSUS_YEARS[-1]

    census = {}
    with open(RAW / "census_travel_to_work.csv") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == latest:
                census[r["lsoa_code"]] = (int(r["workers"]), int(r["cycle_to_work"]))

    zones = []
    for z in read_geojson(RAW / "lsoas.geojson")["features"]:
        p = z["properties"]
        workers, cyclists = census[p["lsoa_code"]]
        zones.append({**z, "properties": {**p, "workers": workers, "cyclists": cyclists,
                                          "census_year": latest}})

    edges = []
    for e in read_geojson(RAW / "edges.geojson")["features"]:
        built = e["properties"]["lane_built_year"]
        edges.append({**e, "properties": {"edge_id": e["properties"]["edge_id"],
                                          "has_lane": built is not None and built <= today}})

    write_geojson(FRONTEND_DATA / "lsoas.geojson", zones)
    write_geojson(FRONTEND_DATA / "edges.geojson", edges)
    print(f"Wrote front-end data to {FRONTEND_DATA}")


if __name__ == "__main__":
    main()
