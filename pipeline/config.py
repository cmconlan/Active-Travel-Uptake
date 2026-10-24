"""Shared settings for the Brighton & Hove active travel demo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FRONTEND_DATA = ROOT / "frontend" / "data"

# --- Study area ------------------------------------------------------------------------
# LSOA 2021 names are "<Local authority> <code>", e.g. "Brighton and Hove 027A"
LA_NAME = "Brighton and Hove"
LSOA_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query"
)
LSOA_PWC_SERVICE = (  # population-weighted centroids: where people live, not the polygon middle
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA_PopCentroids_EW_2021_V4/FeatureServer/0/query"
)
# Pull OSM a little beyond the boundary so features near the edge aren't truncated
STUDY_BUFFER_M = 1000

# --- CRSs ------------------------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"   # what OSM and the web map use
CRS_BNG = "EPSG:27700"    # British National Grid, metres: use for all distance/length work

# --- OSM snapshots ---------------------------------------------------------------------
# Overpass "attic" queries return OSM as it was at this timestamp.
# 2021 is set to Census day; 2015 is the start of the comparison window.
SNAPSHOTS = {
    2015: "2015-01-01T00:00:00Z",
    2021: "2021-03-21T00:00:00Z",
}
# osmnx network type: "bike" keeps everything legally cyclable (roads, cycleways, bridleways)
NETWORK_TYPE = "bike"
# Extra OSM way tags to keep on edges (cycle infrastructure lives in these)
EXTRA_WAY_TAGS = [
    "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
    "bicycle", "segregated", "surface", "lit", "foot",
]
OVERPASS_TIMEOUT_S = 900
# Public Overpass servers get busy. Failed requests (429/5xx, timeouts) are retried with
# exponential backoff: 60 s, 120 s, 240 s ... capped at 30 min, with a little random jitter.
RETRY_MAX_ATTEMPTS = 8
RETRY_BASE_WAIT_S = 60
RETRY_MAX_WAIT_S = 1800
# Polite gap between consecutive successful requests
PAUSE_BETWEEN_QUERIES_S = 30
# Override with OVERPASS_URL=... if you use another instance (it must support attic/[date:] queries)
OVERPASS_URL = "https://overpass-api.de/api"

# --- Level of Traffic Stress (LTS) -----------------------------------------------------
# Adapted from Mekuria, Furth & Nixon (2012) for UK speeds (mph) and OSM tags. See pipeline/lts.py.
# Residential/unclassified/service streets at <=20 mph in mixed traffic: LTS 1 or 2 is a judgement
# call that moves Brighton results a lot. 2 is the cautious choice.
LOCAL_20MPH_LTS = 2
# Speed assumed when maxspeed is missing, by road class (UK default urban limit is 30 mph).
# The same defaults are used for both years so imputation can't create fake change.
DEFAULT_SPEED_MPH = {
    "trunk": 40, "primary": 30, "secondary": 30, "tertiary": 30,
    "unclassified": 30, "residential": 30, "road": 30, "service": 20, "living_street": 10,
}
# 2015 edges with no maxspeed borrow the 2021 value for the same OSM way: a limit tagged later
# was usually already in force, and this stops tagging effort reading as a real change.
BACKFILL_SPEED_FROM = 2021
# Matching 2021 edges to their 2015 counterpart: same node pair, else nearest 2015 edge to the
# 2021 edge's midpoint within this distance. No match = the way wasn't in OSM in 2015.
EDGE_MATCH_TOLERANCE_M = 10

# --- POIs ------------------------------------------------------------------------------
# category -> OSM tag filter (osmnx features_from_polygon format)
POI_TAGS = {
    "bus_stop": {"highway": "bus_stop"},
    "rail_station": {"railway": ["station", "halt"]},
    "grocery": {"shop": ["supermarket", "convenience", "greengrocer"]},
    "school": {"amenity": "school"},
}
# Rail POIs whose name matches this are dropped (Volk's Electric Railway is a seafront attraction,
# not a commuter station)
RAIL_EXCLUDE_NAME_PATTERN = "volk"

# --- LSOA features (pipeline/features.py) ----------------------------------------------
# Routed destinations: nearest k by network distance within cap_m. Bus stops are not routed;
# they enter as a control (bus_stops_400m) because good buses compete with cycling.
DESTINATIONS = {
    "rail_station": {"cap_m": 5000, "k": 3},
    "grocery": {"cap_m": 3000, "k": 3},
    "school": {"cap_m": 3000, "k": 3},
}
# Single fixed destination standing in for the main employment area (routed, no cap)
CITY_CENTRE = {"name": "Brighton Clock Tower", "lat": 50.8246, "lon": -0.1420}
LOW_STRESS_MAX_LTS = 2          # LTS <= this counts as low-stress
ACCESS_TOLERANCE_M = 250        # how far off the low-stress network an origin/destination may sit
DETOUR_CAP = 2.0                # low-stress/shortest length ratio is capped here (and used if no route)
LOCAL_RADIUS_M = 1000           # network radius for local low-stress km
BUS_STOP_RADIUS_M = 400
POI_SNAP_MAX_M = 200            # POIs further than this from the network are dropped
# Fixed routes are found once on the 2021 network. A 2021 edge with no 2015 counterpart didn't
# exist (in OSM) in 2015, so it scores as LTS 4 for 2015: no comfortable option there yet.
UNMATCHED_EDGE_LTS = 4

# --- Synthetic census + model (pipeline/synthetic_census.py, pipeline/train_model.py) -----
# Census travel-to-work counts are SYNTHETIC: generated from the real network features with the
# effects below, so the model has a known answer to recover. They are not real findings.
CENSUS_YEARS = (2011, 2021)
SYNTHETIC_SEED = 2026
# Planted change in cycle-to-work share, 2011 -> 2021, in percentage points.
# d_* terms: per 1 SD of feature change (scaled, not centred, so no change = no effect).
# Level terms (controls, baseline): per 1 SD, centred. See pipeline/model_spec.py.
TRUE_EFFECTS_PP = {
    "intercept": 0.5,                      # town-wide trend
    "d_mean_route_lts": -0.6,              # calmer routes -> more cycling
    "d_share_routes_low_stress": 0.4,
    "d_ls_reachable_pois": 0.8,
    "d_ls_local_km": 0.3,
    "d_ls_reachable_pois_x_base": -0.3,    # gains count for more where low-stress access was poor
    "dist_centre_m": -0.3,                 # weaker trend further from the centre
    "bus_stops_400m": -0.2,                # good buses compete with cycling
}
NOISE_SD_PP = 0.4                          # LSOA-level noise, on top of binomial sampling noise
CV_FOLDS = 5

# --- Front end (pipeline/export_frontend.py) ---------------------------------------------
# Users can upgrade streets inside this box; LSOAs whose population-weighted centroid lies
# inside it get predictions. The exported routing network extends further (see export_frontend).
FRONTEND_AREA = {
    "name": "Central Brighton",
    "description": "Seven Dials to the seafront to Kemptown",
    "bbox": [-0.158, 50.818, -0.112, 50.838],   # lon_min, lat_min, lon_max, lat_max
}
UPGRADE_LTS = 1   # a user-drawn lane is a protected / separated track
# Street names offered as one-click example schemes (only shown if found in the area)
PRESET_STREETS = ["Western Road", "North Street", "Edward Street", "Marine Parade"]
# Shown as "Method and code" on the map; leave empty to hide the link
REPO_URL = ""
