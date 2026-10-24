"""Download 2021 LSOA boundaries for Brighton and Hove from the ONS Open Geography Portal.

Output: data/raw/lsoa_2021.gpkg (EPSG:27700), columns lsoa21cd, lsoa21nm, area_km2
        data/raw/study_area.gpkg (LA boundary + buffer, EPSG:27700), used to clip OSM pulls
        data/raw/lsoa_2021_pwc.gpkg (population-weighted centroids, EPSG:27700): route origins
"""
import geopandas as gpd
import requests

from config import CRS_BNG, CRS_WGS84, LA_NAME, LSOA_PWC_SERVICE, LSOA_SERVICE, RAW, STUDY_BUFFER_M

PAGE_SIZE = 1000


def query_lsoas(la_name: str) -> gpd.GeoDataFrame:
    """Page through the ArcGIS FeatureServer and return matching LSOAs as a GeoDataFrame."""
    frames, offset = [], 0
    while True:
        params = {
            "where": f"LSOA21NM LIKE '{la_name} %'",
            "outFields": "LSOA21CD,LSOA21NM",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        }
        r = requests.get(LSOA_SERVICE, params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        if "error" in page:
            raise RuntimeError(f"ONS service error: {page['error']}")
        feats = page.get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs=CRS_WGS84))
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    if not frames:
        raise RuntimeError(f"No LSOAs returned for '{la_name}'. Check LA_NAME and LSOA_SERVICE in config.py")
    gdf = gpd.pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(gdf, crs=CRS_WGS84)


def tidy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.rename(columns=str.lower).to_crs(CRS_BNG)
    gdf = gdf[["lsoa21cd", "lsoa21nm", "geometry"]].drop_duplicates("lsoa21cd").sort_values("lsoa21cd")
    gdf["area_km2"] = (gdf.area / 1e6).round(4)
    return gdf.reset_index(drop=True)


def study_area(lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary = lsoas.union_all()
    return gpd.GeoDataFrame(
        {"name": [LA_NAME, f"{LA_NAME} + {STUDY_BUFFER_M} m"]},
        geometry=[boundary, boundary.buffer(STUDY_BUFFER_M)],
        crs=CRS_BNG,
    )


def fetch_centroids(lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Population-weighted centroids for our LSOAs. Falls back to a point inside each polygon
    (with a warning) if the ONS service can't be reached, so later stages still run."""
    codes = ", ".join(f"'{c}'" for c in lsoas["lsoa21cd"])
    try:
        r = requests.post(LSOA_PWC_SERVICE, timeout=120, data={
            "where": f"LSOA21CD IN ({codes})", "outFields": "*", "outSR": 4326, "f": "geojson"})
        r.raise_for_status()
        page = r.json()
        if "error" in page or not page.get("features"):
            raise RuntimeError(page.get("error", "no features returned"))
        pwc = gpd.GeoDataFrame.from_features(page["features"], crs=CRS_WGS84).rename(columns=str.lower)
        pwc = pwc[["lsoa21cd", "geometry"]].to_crs(CRS_BNG)
        pwc["centroid_type"] = "population_weighted"
    except Exception as err:  # noqa: BLE001 - any failure -> documented fallback
        print(f"  WARNING: population-weighted centroids unavailable ({err}); using polygon centres")
        pwc = gpd.GeoDataFrame({"lsoa21cd": lsoas["lsoa21cd"]},
                               geometry=lsoas.geometry.representative_point(), crs=CRS_BNG)
        pwc["centroid_type"] = "polygon_point"
    missing = set(lsoas["lsoa21cd"]) - set(pwc["lsoa21cd"])
    if missing:
        raise RuntimeError(f"{len(missing)} LSOAs have no centroid, e.g. {sorted(missing)[:3]}")
    return pwc.merge(lsoas[["lsoa21cd", "lsoa21nm"]], on="lsoa21cd")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    lsoas = tidy(query_lsoas(LA_NAME))
    lsoas.to_file(RAW / "lsoa_2021.gpkg", layer="lsoa_2021", driver="GPKG")
    study_area(lsoas).to_file(RAW / "study_area.gpkg", layer="study_area", driver="GPKG")
    pwc = fetch_centroids(lsoas)
    pwc.to_file(RAW / "lsoa_2021_pwc.gpkg", layer="lsoa_2021_pwc", driver="GPKG")
    print(f"{len(lsoas)} LSOAs, {lsoas.area_km2.sum():.1f} km² -> {RAW / 'lsoa_2021.gpkg'}")


if __name__ == "__main__":
    main()
