"""Download the OSM cycle network and POIs for Brighton and Hove as they were at each snapshot date.

Uses Overpass "attic" queries (the [date:...] setting), so each pull reflects OSM at that
moment, not today. Run fetch_lsoas.py first: the study area polygon comes from it.

Outputs, per snapshot year (EPSG:4326 as downloaded; later stages project to EPSG:27700):
  data/raw/osm_network_<year>.graphml   full graph, for routing-based features
  data/raw/osm_network_<year>.gpkg      nodes + edges layers, for GIS inspection
  data/raw/pois_<year>.gpkg             one point per POI: category, name, osm_type, osm_id

Usage: python pipeline/fetch_osm.py [--years 2015 2021] [--force]

Load on the public Overpass server is kept low: one network query and one combined POI query
per year (4 requests in total), a pause between them, and exponential backoff when the server
is busy. Attic queries are slow, so expect this to take a while. Successful responses are cached
in ./cache and finished outputs are skipped on rerun, so an interrupted run resumes where it stopped.
"""
import argparse
import os
import random
import re
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from config import (CRS_BNG, CRS_WGS84, EXTRA_WAY_TAGS, NETWORK_TYPE, OVERPASS_TIMEOUT_S,
                    OVERPASS_URL, PAUSE_BETWEEN_QUERIES_S, POI_TAGS, RAW, RETRY_BASE_WAIT_S,
                    RETRY_MAX_ATTEMPTS, RETRY_MAX_WAIT_S, ROOT, SNAPSHOTS)

DEFAULT_WAY_TAGS = list(ox.settings.useful_tags_way)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def configure_osmnx(timestamp: str) -> None:
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}]{{maxsize}}[date:"{timestamp}"]'
    ox.settings.overpass_url = os.environ.get("OVERPASS_URL", OVERPASS_URL)
    ox.settings.overpass_rate_limit = True  # osmnx also waits for a free slot via /status
    ox.settings.requests_timeout = OVERPASS_TIMEOUT_S
    ox.settings.useful_tags_way = list(dict.fromkeys(DEFAULT_WAY_TAGS + EXTRA_WAY_TAGS))
    ox.settings.cache_folder = str(ROOT / "cache")
    ox.settings.use_cache = True
    ox.settings.log_console = False


def status_code(err: Exception):
    """Pull the HTTP status out of osmnx's error message, e.g. "'overpass-api.de' responded: 502 ..."."""
    m = re.search(r"responded:\s*(\d{3})", str(err))
    return int(m.group(1)) if m else None


def with_backoff(fn, label: str):
    """Call fn(), retrying busy-server errors with exponential backoff plus jitter."""
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except ResponseStatusCodeError as err:
            code = status_code(err)
            if code not in RETRYABLE_STATUS:
                raise  # e.g. 400 means a malformed query: retrying won't help
            reason = f"HTTP {code}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
            reason = type(err).__name__
        if attempt == RETRY_MAX_ATTEMPTS:
            raise RuntimeError(f"{label}: giving up after {attempt} attempts ({reason}). "
                               "Rerun later: finished outputs are kept.")
        wait = min(RETRY_BASE_WAIT_S * 2 ** (attempt - 1), RETRY_MAX_WAIT_S)
        wait *= random.uniform(0.9, 1.2)
        print(f"  {label}: server busy ({reason}), attempt {attempt}/{RETRY_MAX_ATTEMPTS}; "
              f"retrying in {wait / 60:.1f} min")
        time.sleep(wait)


def study_polygon():
    sa = gpd.read_file(RAW / "study_area.gpkg")
    buffered = sa.iloc[[-1]]  # last row is the buffered boundary
    return buffered.to_crs(CRS_WGS84).geometry.iloc[0]


def fetch_network(polygon, year: int) -> None:
    G = with_backoff(lambda: ox.graph_from_polygon(polygon, network_type=NETWORK_TYPE, simplify=True),
                     f"network {year}")
    ox.save_graphml(G, RAW / f"osm_network_{year}.graphml")
    ox.save_graph_geopackage(G, RAW / f"osm_network_{year}.gpkg")
    print(f"  network {year}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")


def combined_tags() -> dict:
    """Merge every category's tag filter into one dict, so all POIs come back in a single query."""
    merged = {}
    for tags in POI_TAGS.values():
        for key, val in tags.items():
            vals = val if isinstance(val, list) else [val]
            merged[key] = list(dict.fromkeys(merged.get(key, []) + vals))
    return merged


def category_mask(gdf: gpd.GeoDataFrame, tags: dict):
    mask = pd.Series(False, index=gdf.index)
    for key, val in tags.items():
        if key in gdf:
            mask |= gdf[key].isin(val if isinstance(val, list) else [val])
    return mask


def to_points(gdf: gpd.GeoDataFrame, category: str) -> gpd.GeoDataFrame:
    """Collapse any geometry (school grounds, supermarket buildings) to one point inside it."""
    gdf = gdf.reset_index()  # osmnx index is (element, id)
    pts = gdf.to_crs(CRS_BNG).geometry.representative_point().to_crs(CRS_WGS84)
    return gpd.GeoDataFrame(
        {
            "category": category,
            "name": gdf["name"] if "name" in gdf else None,
            "osm_type": gdf["element"],
            "osm_id": gdf["id"].astype("int64"),
        },
        geometry=pts.values,
        crs=CRS_WGS84,
    )


def fetch_pois(polygon, year: int) -> None:
    try:
        raw = with_backoff(lambda: ox.features_from_polygon(polygon, combined_tags()), f"pois {year}")
    except InsufficientResponseError:
        raise RuntimeError(f"pois {year}: Overpass returned no POIs at all; check the study area")
    frames = []
    for category, tags in POI_TAGS.items():
        sub = raw[category_mask(raw, tags)]
        if sub.empty:
            print(f"  pois {year}: no {category} found")
            continue
        frames.append(to_points(sub, category))
    pois = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=CRS_WGS84)
    pois = pois.drop_duplicates(["category", "osm_type", "osm_id"])
    pois.to_file(RAW / f"pois_{year}.gpkg", layer="pois", driver="GPKG")
    counts = ", ".join(f"{k} {v:,}" for k, v in pois.category.value_counts().items())
    print(f"  pois {year}: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(SNAPSHOTS))
    parser.add_argument("--force", action="store_true", help="re-download even if outputs exist")
    args = parser.parse_args()

    polygon = study_polygon()
    first_request = True
    for year in args.years:
        print(f"OSM as of {SNAPSHOTS[year]}")
        configure_osmnx(SNAPSHOTS[year])
        jobs = [
            (RAW / f"osm_network_{year}.graphml", fetch_network, "network"),
            (RAW / f"pois_{year}.gpkg", fetch_pois, "pois"),
        ]
        for output, fetch, name in jobs:
            if output.exists() and not args.force:
                print(f"  {name} {year}: exists, skipping (--force to redo)")
                continue
            if not first_request:
                time.sleep(PAUSE_BETWEEN_QUERIES_S)
            fetch(polygon, year)
            first_request = False


if __name__ == "__main__":
    main()
