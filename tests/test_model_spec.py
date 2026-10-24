"""Tests for the shared model specification. Run with: python -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from model_spec import CHANGE_TERMS, INTERACTION_TERM, TERMS, design, fit_scaler, training_frame  # noqa: E402


def frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in CHANGE_TERMS})
    df["ls_reachable_pois_2015"] = rng.poisson(4, n)
    df["dist_centre_m"] = rng.uniform(300, 7000, n)
    df["bus_stops_400m"] = rng.poisson(6, n)
    return training_frame(df, 2015)


def test_columns_in_term_order():
    df = frame()
    assert list(design(df, fit_scaler(df)).columns) == TERMS


def test_no_network_change_means_no_change_term_contribution():
    """Scenario property the front end relies on: zero change -> change terms and interaction are 0."""
    df = frame()
    scaler = fit_scaler(df)
    still = df.copy()
    still[CHANGE_TERMS] = 0.0
    X = design(still, scaler)
    assert np.allclose(X[CHANGE_TERMS + [INTERACTION_TERM]], 0)


def test_change_terms_are_scaled_not_centred():
    df = frame()
    scaler = fit_scaler(df)
    for c in CHANGE_TERMS:
        assert scaler[c]["mean"] == 0.0
        assert np.allclose(design(df, scaler)[c], df[c] / scaler[c]["sd"])
