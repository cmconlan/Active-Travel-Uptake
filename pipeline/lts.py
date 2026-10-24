"""Assign a Level of Traffic Stress (LTS 1-4) to every edge of the 2015 and 2021 OSM networks.

LTS 1  comfortable for most people, including children (traffic-free or physically separated)
LTS 2  most adults (quiet streets, painted lanes on calm roads)
LTS 3  confident cyclists (busier roads, lanes on main roads, shared bus lanes)
LTS 4  only the "strong and fearless" (main roads in mixed traffic, fast or multi-lane roads)

Rules adapted from Mekuria, Furth & Nixon (2012) for UK speeds and OSM tagging; first match wins.
Edges merged by osmnx simplification can carry lists of tag values: we take the worst case
(highest-stress road class, highest speed, weakest cycle provision).

Outputs: data/processed/edges_lts.gpkg
  layer edges_2015   2015 network with lts, lts_reason, speed_mph, speed_source
  layer edges_2021   2021 network with the same, plus lts_2015 / lts_change / match_2015:
                     each 2021 edge scored as it was in 2015. Features route on this layer.
         data/processed/lts_map.png   2015, 2021 and change maps for eyeballing
"""
import ast
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import (BACKFILL_SPEED_FROM, CRS_BNG, DEFAULT_SPEED_MPH, EDGE_MATCH_TOLERANCE_M,
                    LOCAL_20MPH_LTS, PROCESSED, RAW, SNAPSHOTS)

# Road classes, ordered from least to most stressful (used to pick the worst of a list)
HIGHWAY_CLASS = {
    "cycleway": "cycle", "path": "path", "footway": "path", "bridleway": "path", "steps": "path",
    "track": "path", "pedestrian": "pedestrian", "living_street": "living", "service": "service",
    "residential": "local", "unclassified": "local", "road": "local",
    "tertiary": "collector",
    "secondary": "arterial", "primary": "arterial", "trunk": "arterial", "busway": "arterial",
}
CLASS_ORDER = ["cycle", "path", "pedestrian", "living", "service", "local", "collector", "arterial"]
CYCLING_ALLOWED = {"designated", "yes", "permissive"}
SIDE_TAGS = ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]

# On-road provision, strongest first. "shared_lane" (sharrows) and "separate" (the cycle track is
# mapped as its own way) count as no provision on this edge.
NO_PROVISION, BUS_LANE, PAINTED_LANE, TRACK = 0, 1, 2, 3
PROVISION = {
    "track": TRACK, "opposite_track": TRACK,
    "lane": PAINTED_LANE, "opposite_lane": PAINTED_LANE,
    "share_busway": BUS_LANE, "opposite_share_busway": BUS_LANE, "busway": BUS_LANE,
}
NATIONAL_LIMITS = {"gb:nsl_single": 60, "gb:nsl_dual": 70, "gb:motorway": 70, "gb:zone20": 20}


# --- tag parsing ----------------------------------------------------------------------
def as_list(value) -> list[str]:
    """OSM values arrive as None, a string, or a stringified list ("['lane', 'no']")."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if s.startswith("["):
        try:
            return [str(v).strip().lower() for v in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    return [p.strip().lower() for p in s.split(";") if p.strip()]


def parse_speed_mph(value) -> float | None:
    """Highest speed in the value. UK limits are mph; a bare number is almost always a mph
    limit tagged without its unit, unless it is too high to be one (then treat as km/h)."""
    speeds = []
    for v in as_list(value):
        if v in NATIONAL_LIMITS:
            speeds.append(NATIONAL_LIMITS[v])
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?$", v)
        if not m:
            continue  # "signals", "variable", "none"...
        n, unit = float(m.group(1)), m.group(2)
        if unit in ("km/h", "kmh") or (unit is None and n > 70):
            n *= 0.621
        speeds.append(n)
    return max(speeds) if speeds else None


def parse_lanes(value) -> int | None:
    nums = [int(float(v)) for v in as_list(value) if re.match(r"^\d+(\.\d+)?$", v)]
    return max(nums) if nums else None


def road_class(highway) -> str:
    classes = [HIGHWAY_CLASS.get(h.removesuffix("_link"), "local") for h in as_list(highway)]
    return max(classes, key=CLASS_ORDER.index) if classes else "local"


def provision_level(tags: dict, oneway: bool) -> int:
    """Strength of on-road cycle provision. Two-way roads take the worse side (you have to ride
    both ways); one-way roads take the better side (only one side carries cycle traffic)."""
    def side(name):
        for key in (f"cycleway:{name}", "cycleway:both", "cycleway"):
            vals = as_list(tags.get(key))
            if vals:
                return min(PROVISION.get(v, NO_PROVISION) for v in vals)
        return NO_PROVISION
    left, right = side("left"), side("right")
    return max(left, right) if oneway else min(left, right)


def default_speed(cls_highway) -> float:
    hws = [h.removesuffix("_link") for h in as_list(cls_highway)]
    speeds = [DEFAULT_SPEED_MPH.get(h, 30) for h in hws]
    return max(speeds) if speeds else 30


# --- the rules ------------------------------------------------------------------------
def classify(tags: dict, speed_mph: float) -> tuple[int, str]:
    """Return (LTS, reason) for one edge. `tags` holds raw OSM values; `speed_mph` is resolved."""
    cls = road_class(tags.get("highway"))
    lanes = parse_lanes(tags.get("lanes")) or 0
    oneway = str(tags.get("oneway")).lower() in ("true", "yes", "1")

    # Off-road and traffic-free
    if cls == "cycle":
        return 1, "cycleway"
    if cls in ("path", "pedestrian"):
        if set(as_list(tags.get("bicycle"))) & CYCLING_ALLOWED:
            return 1, "shared path / pedestrian street, cycling allowed"
        return 2, "path, cycling not designated"
    if cls == "living":
        return 1, "living street"

    # On-road provision
    prov = provision_level(tags, oneway)
    if prov == TRACK:
        return 1, "separated cycle track"
    if prov == PAINTED_LANE:
        if speed_mph >= 50:
            return 4, "painted lane, 50+ mph"
        if cls == "arterial" or speed_mph >= 40 or lanes >= 4:
            return 3, "painted lane on main or fast road"
        return 2, "painted lane on calm road"
    if prov == BUS_LANE:
        return (4, "shared bus lane, 40+ mph") if speed_mph >= 40 else (3, "shared bus lane")

    # Mixed traffic
    if speed_mph >= 40 or lanes >= 4:
        return 4, "mixed traffic, 40+ mph or 4+ lanes"
    if cls == "arterial":
        return 4, "main road, no cycle provision"
    if cls == "collector":
        return 3, "tertiary road, mixed traffic"
    if cls == "service":
        return 2, "service road"
    if speed_mph <= 20:
        return LOCAL_20MPH_LTS, "local street, 20 mph"
    return 3, "local street, 30 mph"


# --- applying it to a network ---------------------------------------------------------
def osmids(value) -> list[int]:
    out = []
    for v in as_list(value):
        try:
            out.append(int(float(v)))
        except ValueError:
            pass
    return out


def tagged_speed_lookup(edges: gpd.GeoDataFrame) -> dict[int, float]:
    """OSM way id -> tagged speed (mph), used to backfill another year's missing speeds."""
    lookup = {}
    for ids, raw in zip(edges["osmid"], edges.get("maxspeed", pd.Series(None, index=edges.index))):
        s = parse_speed_mph(raw)
        if s is not None:
            for i in osmids(ids):
                lookup[i] = max(s, lookup.get(i, 0))
    return lookup


def score_network(edges: gpd.GeoDataFrame, backfill: dict | None = None) -> gpd.GeoDataFrame:
    edges = edges.copy()
    speeds, sources, ltss, reasons = [], [], [], []
    cols = [c for c in ["osmid", "maxspeed", "highway", "bicycle", "lanes", "oneway", *SIDE_TAGS] if c in edges]
    for values in edges[cols].itertuples(index=False, name=None):
        tags = dict(zip(cols, values))
        speed, source = parse_speed_mph(tags.get("maxspeed")), "tagged"
        if speed is None and backfill:
            borrowed = [backfill[i] for i in osmids(tags["osmid"]) if i in backfill]
            if borrowed:
                speed, source = max(borrowed), f"backfilled_{BACKFILL_SPEED_FROM}"
        if speed is None:
            speed, source = default_speed(tags.get("highway")), "default"
        lts, reason = classify(tags, speed)
        speeds.append(round(speed, 1)); sources.append(source); ltss.append(lts); reasons.append(reason)
    edges["speed_mph"], edges["speed_source"] = speeds, sources
    edges["lts"], edges["lts_reason"] = ltss, reasons
    edges["length_m"] = edges.length.round(1)
    return edges


def match_to_earlier(later: gpd.GeoDataFrame, earlier: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Give each later edge the LTS of the same street in the earlier network, as `lts_earlier`.
    1) same node pair (OSM node ids are stable), 2) nearest earlier edge to the midpoint."""
    later = later.copy()
    pair = lambda df: [tuple(sorted(p)) for p in zip(df["u"], df["v"])]
    by_pair = pd.Series(earlier["lts"].values, index=pair(earlier)).groupby(level=0).max()
    later["lts_earlier"] = [by_pair.get(p, np.nan) for p in pair(later)]
    later["match_earlier"] = np.where(later["lts_earlier"].notna(), "node_pair", None)

    todo = later["lts_earlier"].isna()
    if todo.any():
        mids = gpd.GeoDataFrame(geometry=later.loc[todo].geometry.interpolate(0.5, normalized=True),
                                crs=later.crs)
        near = gpd.sjoin_nearest(mids, earlier[["lts", "geometry"]], how="left",
                                 max_distance=EDGE_MATCH_TOLERANCE_M)
        near = near.groupby(level=0)["lts"].max()  # ties: take the worse
        later.loc[near.index, "lts_earlier"] = near
        later.loc[near.index[near.notna()], "match_earlier"] = "nearest"
    later["match_earlier"] = later["match_earlier"].fillna("not_in_earlier")
    later["lts_change"] = later["lts"] - later["lts_earlier"]  # negative = less stressful now
    return later


# --- reporting ------------------------------------------------------------------------
LTS_COLOURS = {1: "#1a9850", 2: "#91cf60", 3: "#fc8d59", 4: "#d73027"}


def summarise(edges: gpd.GeoDataFrame, year: int) -> None:
    km = edges.groupby("lts")["length_m"].sum() / 1000
    total = km.sum()
    print(f"\n{year}: {total:,.0f} km of network")
    for lts in (1, 2, 3, 4):
        print(f"  LTS {lts}: {km.get(lts, 0):7,.1f} km  ({km.get(lts, 0) / total:5.1%})")
    src = edges.groupby("speed_source")["length_m"].sum() / edges["length_m"].sum()
    print("  speed source: " + ", ".join(f"{k} {v:.0%}" for k, v in src.items()))
    reasons = (edges.groupby(["lts", "lts_reason"])["length_m"].sum() / 1000).round(1)
    print("  km by rule:\n" + "\n".join(f"    LTS {l}  {r:<50} {v:8,.1f}" for (l, r), v in reasons.items()))


def summarise_change(later: gpd.GeoDataFrame, y0: int, y1: int) -> None:
    km = lambda m: later.loc[m, "length_m"].sum() / 1000
    print(f"\n{y0} -> {y1}, measured on the {y1} network:")
    print(f"  less stressful {km(later.lts_change < 0):,.1f} km | more stressful {km(later.lts_change > 0):,.1f} km | "
          f"unchanged {km(later.lts_change == 0):,.1f} km | not in OSM in {y0} {km(later.match_earlier == 'not_in_earlier'):,.1f} km")
    print("  matched by: " + ", ".join(f"{k} {v:,}" for k, v in later.match_earlier.value_counts().items()))


def plot(earlier, later, lsoas, y0: int, y1: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 24))
    for ax, edges, title in ((axes[0], earlier, f"LTS, OSM {y0}"), (axes[1], later, f"LTS, OSM {y1}")):
        lsoas.boundary.plot(ax=ax, color="#dddddd", linewidth=0.3)
        for lts in (4, 3, 2, 1):  # calm streets drawn last, on top
            if (edges.lts == lts).any():
                edges[edges.lts == lts].plot(ax=ax, color=LTS_COLOURS[lts], linewidth=1.4 if lts == 1 else 0.6)
        ax.legend(handles=[Line2D([], [], color=c, lw=3, label=f"LTS {l}") for l, c in LTS_COLOURS.items()],
                  loc="lower right", frameon=False)
        ax.set_title(title)

    ax = axes[2]
    lsoas.boundary.plot(ax=ax, color="#dddddd", linewidth=0.3)
    cats = [
        ("unchanged", later.lts_change == 0, "#d0d0d0", 0.4),
        (f"not in OSM in {y0}", later.match_earlier == "not_in_earlier", "#8e6bbf", 0.8),
        ("more stressful", later.lts_change > 0, "#b2182b", 1.4),
        ("less stressful", later.lts_change < 0, "#2166ac", 1.4),
    ]
    for _, mask, colour, lw in cats:
        if mask.any():
            later[mask].plot(ax=ax, color=colour, linewidth=lw)
    ax.legend(handles=[Line2D([], [], color=c, lw=3, label=l) for l, _, c, _ in cats], loc="lower right", frameon=False)
    ax.set_title(f"Change in LTS, {y0} to {y1}")

    minx, miny, maxx, maxy = lsoas.total_bounds
    for ax in axes:
        ax.set_xlim(minx - 300, maxx + 300); ax.set_ylim(miny - 300, maxy + 300); ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PROCESSED / "lts_map.png", dpi=150)
    plt.close(fig)
    print(f"\nMap -> {PROCESSED / 'lts_map.png'}")


def main():
    y0, y1 = sorted(SNAPSHOTS)
    load = lambda y: gpd.read_file(RAW / f"osm_network_{y}.gpkg", layer="edges").to_crs(CRS_BNG)
    raw0, raw1 = load(y0), load(y1)

    later = score_network(raw1)
    backfill = tagged_speed_lookup(raw1) if BACKFILL_SPEED_FROM == y1 else None
    earlier = score_network(raw0, backfill=backfill)
    later = match_to_earlier(later, earlier)

    summarise(earlier, y0)
    summarise(later, y1)
    summarise_change(later, y0, y1)
    plot(earlier, later, gpd.read_file(RAW / "lsoa_2021.gpkg"), y0, y1)

    base = ["u", "v", "key", "osmid", "name", "highway", "length_m", "speed_mph", "speed_source", "lts_reason"]
    out0 = earlier[[c for c in base if c in earlier] + ["lts", "geometry"]]
    out1 = later[[c for c in base if c in later] + ["lts", "lts_earlier", "lts_change", "match_earlier", "geometry"]]
    out1 = out1.rename(columns={"lts": f"lts_{y1}", "lts_earlier": f"lts_{y0}", "match_earlier": f"match_{y0}"})
    out = PROCESSED / "edges_lts.gpkg"
    out0.to_file(out, layer=f"edges_{y0}", driver="GPKG")
    out1.to_file(out, layer=f"edges_{y1}", driver="GPKG")
    print(f"Wrote {out} (layers edges_{y0}, edges_{y1})")


if __name__ == "__main__":
    main()
