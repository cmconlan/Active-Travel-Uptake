"""LSOA-level route features, measured in 2015 and 2021 and differenced.

Everything is computed on ONE network geometry (2021) scored with each year's LTS, so a change in
a feature reflects a change in traffic stress, not in route choice or in which POIs exist:
  * routes: shortest paths from each LSOA's population-weighted centroid to its nearest k rail
    stations, grocery shops and schools (within a cap), plus the city centre. Found once, scored twice.
  * destinations: the 2021 POI set, used for both years.
  * low-stress network for year y: edges with LTS_y <= LOW_STRESS_MAX_LTS. A 2021 edge that wasn't in
    OSM in 2015 is not part of the 2015 low-stress network, and scores UNMATCHED_EDGE_LTS on 2015 routes.

Features (each as <name>_2015, <name>_2021 and d_<name> = 2021 - 2015):
  mean_route_lts           mean over routes of length-weighted LTS                 (lower = better)
  share_routes_low_stress  share of routes whose WORST edge is low-stress           (higher = better)
  low_stress_route_share   share of total route length that is low-stress           (higher = better)
  ls_detour                low-stress route length / shortest, capped at DETOUR_CAP (lower = better)
  ls_reachable_pois        POIs reachable within their cap using only low-stress edges (higher = better)
  ls_island_km             size of the connected low-stress network around the origin (higher = better)
  ls_local_km              low-stress km within LOCAL_RADIUS_M network distance       (higher = better)
Controls (not differenced): dist_centre_m, bus_stops_400m, n_routes.

Outputs (data/processed/):
  lsoa_features.csv / .gpkg   one row per LSOA
  routes.csv                  one row per route, with its edge ids (row index of edges_2021 in
                              edges_lts.gpkg): the front end re-scores these when lanes are added
  features_map.png            changes in four headline features
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from config import (ACCESS_TOLERANCE_M, BUS_STOP_RADIUS_M, CITY_CENTRE, CRS_BNG, CRS_WGS84,
                    DESTINATIONS, DETOUR_CAP, LOCAL_RADIUS_M, LOW_STRESS_MAX_LTS, POI_SNAP_MAX_M,
                    PROCESSED, RAIL_EXCLUDE_NAME_PATTERN, RAW, SNAPSHOTS, UNMATCHED_EDGE_LTS)

Y0, Y1 = sorted(SNAPSHOTS)
YEARS = (Y0, Y1)
MAX_CAP = max(d["cap_m"] for d in DESTINATIONS.values())
FEATURES = ["mean_route_lts", "share_routes_low_stress", "low_stress_route_share", "ls_detour",
            "ls_reachable_pois", "ls_island_km", "ls_local_km"]
LOWER_IS_BETTER = {"mean_route_lts", "ls_detour"}


# --- network --------------------------------------------------------------------------
class Snapper:
    """Nearest graph node to a point, within a maximum distance."""
    def __init__(self, nodes: list, coords: dict):
        self.nodes = list(nodes)
        self.tree = cKDTree(np.array([coords[n] for n in self.nodes])) if self.nodes else None

    def __call__(self, xy, max_d=np.inf):
        if self.tree is None:
            return None, np.inf
        d, i = self.tree.query(xy, distance_upper_bound=max_d)
        return (self.nodes[i], float(d)) if np.isfinite(d) else (None, np.inf)


def undirected_graph(edges: gpd.GeoDataFrame, keep: np.ndarray) -> nx.Graph:
    """Undirected graph over the kept edges; parallel edges collapse to the shortest.
    (Cycling one-way restrictions are ignored: a simplification worth stating.)"""
    G = nx.Graph()
    sub = edges.loc[keep]
    for eid, u, v, length in zip(sub.index, sub["u"], sub["v"], sub["length_m"]):
        if u == v or (G.has_edge(u, v) and G[u][v]["length"] <= length):
            continue
        G.add_edge(u, v, length=float(length), eid=int(eid))
    return G


class Network:
    def __init__(self, edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame):
        self.edges = edges.reset_index(drop=True)
        self.length = self.edges["length_m"].to_numpy(float)
        raw = {y: self.edges[f"lts_{y}"].to_numpy(float) for y in YEARS}
        # route scoring: unmatched (NaN) 2015 edges score UNMATCHED_EDGE_LTS
        self.lts = {y: np.where(np.isnan(raw[y]), UNMATCHED_EDGE_LTS, raw[y]) for y in YEARS}
        # low-stress membership: NaN compares False, so unmatched edges are excluded
        self.low = {y: raw[y] <= LOW_STRESS_MAX_LTS for y in YEARS}

        self.coords = dict(zip(nodes["osmid"], zip(nodes.geometry.x, nodes.geometry.y)))
        self.G = undirected_graph(self.edges, np.ones(len(self.edges), bool))
        self.snap_full = Snapper(self.G.nodes, self.coords)
        self.H, self.snap_ls, self.comp_of, self.comp_km = {}, {}, {}, {}
        for y in YEARS:
            H = undirected_graph(self.edges, self.low[y])
            self.H[y], self.snap_ls[y] = H, Snapper(H.nodes, self.coords)
            self.comp_of[y], self.comp_km[y] = {}, {}
            for cid, comp in enumerate(nx.connected_components(H)):
                self.comp_km[y][cid] = H.subgraph(comp).size(weight="length") / 1000
                for n in comp:
                    self.comp_of[y][n] = cid

    def path_eids(self, path: list) -> list[int]:
        return [self.G[a][b]["eid"] for a, b in zip(path, path[1:])]


# --- inputs ---------------------------------------------------------------------------
def load_inputs():
    edges = gpd.read_file(PROCESSED / "edges_lts.gpkg", layer=f"edges_{Y1}")
    nodes = gpd.read_file(RAW / f"osm_network_{Y1}.gpkg", layer="nodes").to_crs(CRS_BNG)
    lsoas = gpd.read_file(RAW / "lsoa_2021.gpkg")
    pwc_path = RAW / "lsoa_2021_pwc.gpkg"
    if pwc_path.exists():
        origins = gpd.read_file(pwc_path)
    else:
        print("WARNING: no population-weighted centroids (rerun fetch_lsoas.py); using polygon points")
        origins = gpd.GeoDataFrame(lsoas[["lsoa21cd", "lsoa21nm"]], geometry=lsoas.representative_point(),
                                   crs=CRS_BNG).assign(centroid_type="polygon_point")
    pois = gpd.read_file(RAW / f"pois_{Y1}.gpkg").to_crs(CRS_BNG)
    return edges, nodes, lsoas, origins.to_crs(CRS_BNG), pois


def prepare_destinations(pois: gpd.GeoDataFrame, net: Network) -> pd.DataFrame:
    """Routed destinations snapped to the full network and to each year's low-stress network."""
    rail = pois["category"] == "rail_station"
    excluded = rail & pois["name"].fillna("").str.lower().str.contains(RAIL_EXCLUDE_NAME_PATTERN)
    dest = pois[pois["category"].isin(DESTINATIONS) & ~excluded].copy()
    centre = gpd.GeoSeries([Point(CITY_CENTRE["lon"], CITY_CENTRE["lat"])], crs=CRS_WGS84).to_crs(CRS_BNG)
    dest = pd.concat([dest, gpd.GeoDataFrame({"category": ["city_centre"], "name": [CITY_CENTRE["name"]],
                                              "osm_type": ["point"], "osm_id": [0]},
                                             geometry=centre.values, crs=CRS_BNG)], ignore_index=True)
    dest["dest_id"] = dest["osm_type"].astype(str) + "/" + dest["osm_id"].astype(str)
    dest["x"], dest["y"] = dest.geometry.x, dest.geometry.y
    xy = list(zip(dest["x"], dest["y"]))
    dest["node"], dest["snap_m"] = zip(*(net.snap_full(p, POI_SNAP_MAX_M) for p in xy))
    for y in YEARS:
        dest[f"ls_node_{y}"], dest[f"ls_snap_{y}"] = zip(*(net.snap_ls[y](p, ACCESS_TOLERANCE_M) for p in xy))
    dropped = dest["node"].isna()
    if dropped.any():
        print(f"  dropped {dropped.sum()} destinations more than {POI_SNAP_MAX_M} m from the network")
    dest = dest[~dropped].drop_duplicates(["category", "node"])
    if "city_centre" not in set(dest["category"]):
        raise RuntimeError("City centre could not be snapped to the network: check CITY_CENTRE")
    print("  destinations: " + ", ".join(f"{k} {v}" for k, v in dest["category"].value_counts().items()))
    return pd.DataFrame(dest.drop(columns="geometry")).reset_index(drop=True)


# --- per-origin computation -----------------------------------------------------------
def find_routes(o_node, net: Network, dest: pd.DataFrame) -> tuple[list[dict], dict]:
    """Fixed routes from one origin, found on the full 2021 network."""
    dist, paths = nx.single_source_dijkstra(net.G, o_node, cutoff=MAX_CAP, weight="length")
    routes = []
    for cat, spec in DESTINATIONS.items():
        cands = dest[(dest["category"] == cat) & dest["node"].isin(dist.keys())].copy()
        cands["d"] = cands["node"].map(dist)
        cands = cands[(cands["d"] <= spec["cap_m"]) & (cands["d"] > 0)].nsmallest(spec["k"], "d")
        for rank, row in enumerate(cands.itertuples(), 1):
            routes.append({"category": cat, "rank": rank, "dest_idx": row.Index, "dest_id": row.dest_id,
                           "length_m": row.d, "eids": net.path_eids(paths[row.node])})
    c = dest.index[dest["category"] == "city_centre"][0]
    try:
        length, path = nx.bidirectional_dijkstra(net.G, o_node, dest.at[c, "node"], weight="length")
        if length > 0:
            routes.append({"category": "city_centre", "rank": 1, "dest_idx": c, "dest_id": dest.at[c, "dest_id"],
                           "length_m": length, "eids": net.path_eids(path)})
    except nx.NetworkXNoPath:
        pass
    return routes, dist


def local_edge_ids(dist: dict, net: Network) -> list[int]:
    """Edges with both ends within LOCAL_RADIUS_M network distance of the origin."""
    local_nodes = [n for n, d in dist.items() if d <= LOCAL_RADIUS_M]
    return [eid for _, _, eid in net.G.subgraph(local_nodes).edges(data="eid")]


def score_year(y, o_xy, routes, dist, net: Network, dest: pd.DataFrame) -> dict:
    """All year-y features for one origin; also writes per-route metrics into `routes`."""
    o_ls, o_snap = net.snap_ls[y](o_xy, ACCESS_TOLERANCE_M)
    longest = max((r["length_m"] for r in routes), default=0)
    ls_dist = {}
    if o_ls is not None:
        ls_dist = nx.single_source_dijkstra_path_length(
            net.H[y], o_ls, cutoff=max(MAX_CAP, DETOUR_CAP * longest), weight="length")

    def ls_length(di):
        node = dest.at[di, f"ls_node_{y}"]
        if o_ls is None or node is None or node not in ls_dist:
            return np.inf
        return o_snap + ls_dist[node] + dest.at[di, f"ls_snap_{y}"]

    for r in routes:
        e = np.asarray(r["eids"], int)
        lts, length = net.lts[y][e], net.length[e]
        r[f"mean_lts_{y}"] = float(np.average(lts, weights=length))
        r[f"max_lts_{y}"] = float(lts.max())
        r[f"low_stress_m_{y}"] = float(length[lts <= LOW_STRESS_MAX_LTS].sum())
        r[f"ls_detour_{y}"] = float(np.clip(ls_length(r["dest_idx"]) / r["length_m"], 1.0, DETOUR_CAP))

    reach = {}
    for cat, spec in DESTINATIONS.items():
        idx = dest.index[dest["category"] == cat]
        reach[cat] = int(sum(ls_length(i) <= spec["cap_m"] for i in idx))

    local_eids = local_edge_ids(dist, net)
    local_km = net.length[local_eids][net.low[y][local_eids]].sum() / 1000 if local_eids else 0.0

    total_len = sum(r["length_m"] for r in routes)
    feats = {
        "mean_route_lts": np.mean([r[f"mean_lts_{y}"] for r in routes]) if routes else np.nan,
        "share_routes_low_stress": np.mean([r[f"max_lts_{y}"] <= LOW_STRESS_MAX_LTS for r in routes]) if routes else np.nan,
        "low_stress_route_share": sum(r[f"low_stress_m_{y}"] for r in routes) / total_len if routes else np.nan,
        "ls_detour": np.mean([r[f"ls_detour_{y}"] for r in routes]) if routes else np.nan,
        "ls_reachable_pois": sum(reach.values()),
        "ls_island_km": net.comp_km[y][net.comp_of[y][o_ls]] if o_ls is not None else 0.0,
        "ls_local_km": local_km,
    }
    feats.update({f"ls_reach_{cat}": n for cat, n in reach.items()})
    return feats


# --- outputs --------------------------------------------------------------------------
def plot_changes(gdf: gpd.GeoDataFrame) -> None:
    shown = ["mean_route_lts", "low_stress_route_share", "ls_reachable_pois", "ls_detour"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, f in zip(axes.flat, shown):
        col = gdf[f"d_{f}"]
        lim = max(abs(col.min()), abs(col.max())) or 1
        # RdBu runs red (low) -> blue (high); flip it when a decrease is the improvement
        cmap = "RdBu_r" if f in LOWER_IS_BETTER else "RdBu"  # blue = better either way
        gdf.plot(column=f"d_{f}", ax=ax, cmap=cmap, vmin=-lim, vmax=lim, legend=True,
                 edgecolor="white", linewidth=0.2, missing_kwds={"color": "#eeeeee"})
        ax.set_title(f"Change in {f}, {Y0} to {Y1}" + ("  (lower is better)" if f in LOWER_IS_BETTER else ""))
        ax.set_axis_off()
    fig.suptitle("Blue = less traffic stress", y=0.99)
    fig.tight_layout()
    fig.savefig(PROCESSED / "features_map.png", dpi=150)
    plt.close(fig)


def main():
    edges, nodes, lsoas, origins, pois = load_inputs()
    print(f"Building networks ({len(edges):,} edges)...")
    net = Network(edges, nodes)
    for y in YEARS:
        print(f"  low-stress network {y}: {net.H[y].size(weight='length') / 1000:,.0f} km, "
              f"{nx.number_connected_components(net.H[y]):,} islands")
    dest = prepare_destinations(pois, net)
    bus = pois[pois["category"] == "bus_stop"]
    bus_tree = cKDTree(np.c_[bus.geometry.x, bus.geometry.y]) if len(bus) else None

    rows, route_rows = [], []
    print(f"Routing from {len(origins)} LSOA centroids...")
    for o in origins.itertuples():
        o_xy = (o.geometry.x, o.geometry.y)
        o_node, o_snap = net.snap_full(o_xy)
        routes, dist = find_routes(o_node, net, dest)
        row = {"lsoa21cd": o.lsoa21cd, "lsoa21nm": o.lsoa21nm, "centroid_type": o.centroid_type,
               "origin_snap_m": round(o_snap, 1), "n_routes": len(routes),
               "dist_centre_m": next((r["length_m"] for r in routes if r["category"] == "city_centre"), np.nan),
               "bus_stops_400m": len(bus_tree.query_ball_point(o_xy, BUS_STOP_RADIUS_M)) if bus_tree else 0}
        for y in YEARS:
            row.update({f"{k}_{y}": v for k, v in score_year(y, o_xy, routes, dist, net, dest).items()})
        rows.append(row)
        for r in routes:
            route_rows.append({"lsoa21cd": o.lsoa21cd, **{k: v for k, v in r.items() if k != "dest_idx"},
                               "eids": " ".join(map(str, r["eids"]))})

    feats = pd.DataFrame(rows)
    for f in FEATURES + [f"ls_reach_{c}" for c in DESTINATIONS]:
        feats[f"d_{f}"] = feats[f"{f}_{Y1}"] - feats[f"{f}_{Y0}"]
    feats = feats.round(4)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_csv(PROCESSED / "lsoa_features.csv", index=False)
    pd.DataFrame(route_rows).round(4).to_csv(PROCESSED / "routes.csv", index=False)
    gdf = lsoas[["lsoa21cd", "geometry"]].merge(feats, on="lsoa21cd")
    gdf.to_file(PROCESSED / "lsoa_features.gpkg", layer="lsoa_features", driver="GPKG")
    plot_changes(gdf)

    d_cols = [f"d_{f}" for f in FEATURES]
    print(f"\nWrote lsoa_features.csv ({len(feats)} LSOAs), routes.csv ({len(route_rows):,} routes), "
          f"lsoa_features.gpkg, features_map.png")
    print("\nFeature levels and change:")
    summary = pd.DataFrame({
        f"mean {Y0}": [feats[f"{f}_{Y0}"].mean() for f in FEATURES],
        f"mean {Y1}": [feats[f"{f}_{Y1}"].mean() for f in FEATURES],
        "mean change": [feats[f"d_{f}"].mean() for f in FEATURES],
        "LSOAs improved": [((feats[f"d_{f}"] < 0) if f in LOWER_IS_BETTER else (feats[f"d_{f}"] > 0)).sum()
                           for f in FEATURES],
    }, index=FEATURES)
    print(summary.round(3).to_string())
    print("\nCorrelation of changes (pick model features that aren't near-duplicates):")
    print(feats[d_cols].corr().round(2).rename(columns=lambda c: c[2:14], index=lambda c: c[2:]).to_string())


if __name__ == "__main__":
    main()
