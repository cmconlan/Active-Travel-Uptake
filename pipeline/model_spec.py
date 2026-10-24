"""The model's terms, defined once and shared by synthetic_census.py, train_model.py and the front end.

  d_share_pp = intercept
             + sum_k  b_k * d_k / sd_k                       (network change terms: scaled, NOT centred,
                                                             so a scenario with no change adds nothing)
             + b_int * (d_reach / sd_reach) * z(reach_base)  (gains matter more where access was poor)
             + sum_c  b_c * z(control_c)                     (level terms: differential town-wide trends)

z(x) = (x - mean) / sd, with means and SDs taken from the training data and stored in model.json.
For training, the baseline is the 2015 value; in a front-end scenario it is the current (2021) value.
A scenario's predicted uplift only involves the change terms, because the controls don't move.
"""
import numpy as np
import pandas as pd

CHANGE_TERMS = ["d_mean_route_lts", "d_share_routes_low_stress", "d_ls_reachable_pois", "d_ls_local_km"]
INTERACTION = ("d_ls_reachable_pois", "ls_reachable_pois_base")
INTERACTION_TERM = "d_ls_reachable_pois_x_base"
CONTROL_TERMS = ["dist_centre_m", "bus_stops_400m"]
TERMS = ["intercept", *CHANGE_TERMS, INTERACTION_TERM, *CONTROL_TERMS]


def fit_scaler(df: pd.DataFrame) -> dict:
    scaler = {}
    for c in CHANGE_TERMS:
        sd = float(df[c].std())
        if not sd > 0:
            raise ValueError(f"{c} has no variation across LSOAs: drop it from CHANGE_TERMS")
        scaler[c] = {"mean": 0.0, "sd": sd}
    for c in [*CONTROL_TERMS, INTERACTION[1]]:
        scaler[c] = {"mean": float(df[c].mean()), "sd": float(df[c].std()) or 1.0}
    return scaler


def design(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    z = lambda c: (df[c] - scaler[c]["mean"]) / scaler[c]["sd"]
    X = pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index)
    for c in CHANGE_TERMS:
        X[c] = z(c)
    X[INTERACTION_TERM] = z(INTERACTION[0]) * z(INTERACTION[1])
    for c in CONTROL_TERMS:
        X[c] = z(c)
    return X[TERMS]


def training_frame(feats: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """LSOA features with the baseline column named as the model expects."""
    out = feats.copy()
    out[INTERACTION[1]] = out[f"ls_reachable_pois_{base_year}"]
    return out
