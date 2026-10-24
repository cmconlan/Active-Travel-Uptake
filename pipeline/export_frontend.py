"""Export everything the central Brighton scenario map needs into one file: frontend/data/scenario.json.

The browser recomputes the model's four network features with the same logic as features.py
(frontend/model.js), for the current network and for the user's scenario, so it needs:
  * a routing network: all edges within MAX_CAP + 500 m of the area, as compact arrays
    (u, v, length, 2021 LTS). Low-stress reachability searches up to MAX_CAP from each origin.
  * street geometry for the area only: the edges users can see and upgrade.
  * per LSOA: its fixed routes (edge indices), its 1 km local edge set, the network nodes it
    could join the low-stress network at (with distances), synthetic workers, and the 2021
    feature values Python computed, so the front end can be checked against Python.
  * destinations near the area with their candidate access nodes.
  * the fitted model (coefficients + scaler).
Run after train_model.py. Check parity with: node tests/frontend_parity.mjs
"""
import json
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

from config import (ACCESS_TOLERANCE_M, CENSUS_YEARS, CRS_BNG, CRS_WGS84, DESTINATIONS,
                    FRONTEND_AREA, FRONTEND_DATA, LOW_STRESS_MAX_LTS, PRESET_STREETS, PROCESSED, REPO_URL,
                    SNAPSHOTS, UPGRADE_LTS)
from features import MAX_CAP, Network, find_routes, load_inputs, local_edge_ids, prepare_destinations
from lts import as_list

Y1 = max(SNAPSHOTS)
PARITY_FEATURES = ["mean_route_lts", "share_routes_low_stress", "ls_reachable_pois", "ls_local_km"]
DISCLAIMER = ("Synthetic data. Streets and traffic-stress ratings come from OpenStreetMap, but the "
              "commuting numbers and the model behind these predictions are synthetic, generated for a "
              "software demo. They do not describe real travel patterns in Brighton and Hove.")


def latlon(geom, nd=6):
    return [[round(y, nd), round(x, nd)] for x, y in geom.coords]


def polygon_latlon(geom):
    polys = getattr(geom, "geoms", [geom])
    return [[latlon(p.exterior, 5)] for p in polys]


def candidates(tree, node_ids, xy, radius):
    """Network nodes within `radius` of a point, nearest first: [[node_index, metres], ...].
    The browser takes the first one that is on the current low-stress network, which is exactly
    what features.py's nearest-low-stress-node snap does."""
    idx = tree.query_ball_point(xy, radius)
    d = np.hypot(tree.data[idx, 0] - xy[0], tree.data[idx, 1] - xy[1]) if idx else []
    order = np.argsort(d, kind="stable")
    return [[int(node_ids[idx[i]]), round(float(d[i]), 2)] for i in order]


def main():
    edges, nodes, lsoas, origins, pois = load_inputs()
    net = Network(edges, nodes)
    dest = prepare_destinations(pois, net)
    model = json.loads((PROCESSED / "model.json").read_text())
    feats = pd.read_csv(PROCESSED / "lsoa_features.csv").set_index("lsoa21cd")
    census = pd.read_csv(PROCESSED / "census_synthetic.csv")
    census = census[census["year"] == CENSUS_YEARS[-1]].set_index("lsoa21cd")

    area = gpd.GeoSeries([box(*FRONTEND_AREA["bbox"])], crs=CRS_WGS84).to_crs(CRS_BNG).iloc[0]
    reach_zone = area.buffer(MAX_CAP + 500)
    e = net.edges
    in_reach = e.intersects(reach_zone).to_numpy()
    in_area = e.intersects(area).to_numpy()

    area_origins = origins[origins.within(area) & origins["lsoa21cd"].isin(census.index)]
    if area_origins.empty:
        raise RuntimeError("No LSOA population-weighted centroids fall inside FRONTEND_AREA")

    # Routes and local edge sets, exactly as features.py builds them
    per_lsoa, needed = {}, set(np.flatnonzero(in_reach))
    for o in area_origins.itertuples():
        o_xy = (o.geometry.x, o.geometry.y)
        o_node, _ = net.snap_full(o_xy)
        routes, dist = find_routes(o_node, net, dest)
        local = local_edge_ids(dist, net)
        per_lsoa[o.lsoa21cd] = (o, o_xy, routes, local)
        needed |= {eid for r in routes for eid in r["eids"]} | set(local)

    # Compact routing network: remap edge and node ids to 0..n-1
    keep = np.array(sorted(needed))
    sub = e.loc[keep]
    node_ids = pd.Index(pd.unique(pd.concat([sub["u"], sub["v"]])))
    e_index = pd.Series(np.arange(len(keep)), index=keep)
    lts_col = f"lts_{Y1}"
    network = {
        "u": node_ids.get_indexer(sub["u"]).tolist(),
        "v": node_ids.get_indexer(sub["v"]).tolist(),
        "len": sub["length_m"].round(2).tolist(),
        "lts": sub[lts_col].astype(int).tolist(),
    }
    coords = np.array([net.coords[n] for n in node_ids])
    tree = cKDTree(coords)
    node_pos = np.arange(len(node_ids))

    def edge_name(raw):
        names = as_list(raw)
        return names[0].title() if names else ""

    streets = []
    for eid in keep[in_area[keep]]:
        geom = gpd.GeoSeries([e.at[eid, "geometry"]], crs=CRS_BNG).to_crs(CRS_WGS84).iloc[0]
        streets.append({"e": int(e_index[eid]), "name": edge_name(e.at[eid, "name"]) if "name" in e else "",
                        "coords": latlon(geom)})

    dests = []
    near = reach_zone.buffer(ACCESS_TOLERANCE_M)
    for d in dest[dest["category"].isin(DESTINATIONS)].itertuples():
        pt = gpd.points_from_xy([d.x], [d.y], crs=CRS_BNG)[0]
        if near.contains(pt):
            dests.append({"cat": d.category, "cand": candidates(tree, node_pos, (d.x, d.y), ACCESS_TOLERANCE_M)})

    lsoa_geom = lsoas.set_index("lsoa21cd").to_crs(CRS_WGS84).geometry
    out_lsoas = []
    for code, (o, o_xy, routes, local) in per_lsoa.items():
        out_lsoas.append({
            "code": code, "name": o.lsoa21nm,
            "workers": int(census.at[code, "workers"]), "cyclists": int(census.at[code, "cycle_to_work"]),
            "dist_centre_m": round(float(feats.at[code, "dist_centre_m"]), 1),
            "bus_stops_400m": int(feats.at[code, "bus_stops_400m"]),
            "origin_cand": candidates(tree, node_pos, o_xy, ACCESS_TOLERANCE_M),
            "routes": [{"cat": r["category"], "len": round(float(r["length_m"]), 2),
                        "e": e_index.loc[r["eids"]].tolist()} for r in routes],
            "local": e_index.loc[local].tolist(),
            "expected": {f: float(feats.at[code, f"{f}_{Y1}"]) for f in PARITY_FEATURES},
            "geometry": polygon_latlon(lsoa_geom[code]),
        })

    presets = []
    for name in PRESET_STREETS:
        hits = [s["e"] for s in streets if name.lower() in s["name"].lower() and network["lts"][s["e"]] > UPGRADE_LTS]
        if hits:
            presets.append({"name": name, "edges": hits})

    scenario = {
        "meta": {"area": FRONTEND_AREA["name"], "area_description": FRONTEND_AREA["description"],
                 "bbox": FRONTEND_AREA["bbox"], "osm_snapshot": SNAPSHOTS[Y1][:10],
                 "census_year": CENSUS_YEARS[-1], "generated": date.today().isoformat(),
                 "disclaimer": DISCLAIMER, "repo_url": REPO_URL},
        "params": {"low_stress_max_lts": LOW_STRESS_MAX_LTS, "upgrade_lts": UPGRADE_LTS, "max_cap_m": MAX_CAP,
                   "caps": {c: s["cap_m"] for c, s in DESTINATIONS.items()}},
        "model": {k: model[k] for k in ("coefficients", "scaler", "change_terms", "interaction", "r2", "cv_r2", "n")},
        "network": network,
        "streets": streets,
        "dests": dests,
        "lsoas": out_lsoas,
        "presets": presets,
    }
    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
    out = FRONTEND_DATA / "scenario.json"
    out.write_text(json.dumps(scenario, separators=(",", ":")))
    mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({mb:.1f} MB): {len(out_lsoas)} LSOAs, {len(streets):,} clickable-area streets, "
          f"{len(keep):,} routing edges, {len(dests)} destinations, presets: "
          f"{', '.join(p['name'] for p in presets) or 'none found'}")
    if mb > 15:
        print("  WARNING: large file; consider a smaller FRONTEND_AREA")


if __name__ == "__main__":
    main()
