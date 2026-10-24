"""Sanity-check the raw downloads: print a summary and save a 2015 vs 2021 map.

Output: data/raw/inspect_raw.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt

from config import CRS_BNG, POI_TAGS, RAW, SNAPSHOTS

CYCLE_TAG_VALUES = ("lane", "track", "share_busway", "shared_lane")
POI_COLOURS = {"bus_stop": "#1f77b4", "rail_station": "#d62728", "grocery": "#2ca02c", "school": "#9467bd"}


def is_cycle_infra(edges: gpd.GeoDataFrame):
    """Rough flag for dedicated or marked cycle provision. Values can be stringified lists after
    osmnx simplification (e.g. "['residential', 'cycleway']"), hence the substring matching."""
    def col(name):
        return edges[name].astype(str) if name in edges else gpd.pd.Series("", index=edges.index)

    flag = col("highway").str.contains("cycleway") | col("bicycle").str.contains("designated")
    for c in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        flag |= col(c).str.contains("|".join(CYCLE_TAG_VALUES))
    return flag


def load(year):
    edges = gpd.read_file(RAW / f"osm_network_{year}.gpkg", layer="edges").to_crs(CRS_BNG)
    edges["cycle_infra"] = is_cycle_infra(edges)
    pois = gpd.read_file(RAW / f"pois_{year}.gpkg").to_crs(CRS_BNG)
    return edges, pois


def main():
    lsoas = gpd.read_file(RAW / "lsoa_2021.gpkg")
    print(f"LSOAs: {len(lsoas)} (expect ~165 for Brighton and Hove), "
          f"{lsoas.area_km2.sum():.1f} km², CRS {lsoas.crs.to_string()}")

    years = list(SNAPSHOTS)
    fig, axes = plt.subplots(1, len(years), figsize=(8 * len(years), 7), sharex=True, sharey=True)
    for ax, year in zip(axes, years):
        edges, pois = load(year)
        km = edges.length.sum() / 1000
        cyc = edges.loc[edges.cycle_infra].length.sum() / 1000
        print(f"{year}: {len(edges):,} edges, {km:,.0f} km network, {cyc:,.1f} km cycle infra "
              f"({cyc / km:.1%}); POIs " + ", ".join(
                  f"{c} {int((pois.category == c).sum())}" for c in POI_TAGS))

        lsoas.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.4)
        edges.loc[~edges.cycle_infra].plot(ax=ax, color="#cfcfcf", linewidth=0.3)
        edges.loc[edges.cycle_infra].plot(ax=ax, color="#e0572b", linewidth=1.2)
        for cat, colour in POI_COLOURS.items():
            sub = pois[pois.category == cat]
            if len(sub):
                sub.plot(ax=ax, color=colour, markersize=3 if cat == "bus_stop" else 10, label=cat)
        ax.set_title(f"OSM as of {SNAPSHOTS[year][:10]}: {cyc:,.0f} km cycle infra")
        ax.set_axis_off()
    axes[-1].legend(loc="lower right", markerscale=2, frameon=False)
    minx, miny, maxx, maxy = lsoas.total_bounds
    axes[0].set_xlim(minx - 500, maxx + 500)
    axes[0].set_ylim(miny - 500, maxy + 500)
    fig.tight_layout()
    out = RAW / "inspect_raw.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Map -> {out}")


if __name__ == "__main__":
    main()
