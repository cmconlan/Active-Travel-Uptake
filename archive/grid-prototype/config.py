"""Shared settings for the Demoton demo. Everything is in a local metric grid (metres)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FRONTEND_DATA = ROOT / "frontend" / "data"

SEED = 42

# Town geometry: 6 x 6 LSOA-like zones, each 400 m square -> 2.4 km town
ZONES_PER_SIDE = 6
ZONE_SIZE_M = 400
# Street grid: a node every 200 m
STREET_SPACING_M = 200

CENSUS_YEARS = (2011, 2021)

# Exposure decay: how quickly a bike lane's influence fades with distance from a zone centroid.
# Must match DECAY_M in frontend/app.js (it is also written into model.json).
DECAY_M = 600

# "True" effect used to generate dummy census data (the model should roughly recover these)
TRUE_BETA_EXPOSURE = 0.004        # change in cycle share per unit exposure
TRUE_BETA_EXPOSURE_SHORT = 0.012  # extra effect scaled by share of short commutes
BACKGROUND_TREND = 0.003          # town-wide drift in cycle share 2011->2021
NOISE_SD = 0.004
