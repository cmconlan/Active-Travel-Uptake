"""Fit a simple first-difference model of cycle-commute uptake on bike-lane exposure change.

  d_cycle_share = b0 + b1 * d_exposure + b2 * d_exposure * short_commute_share + error

b0 absorbs the town-wide trend; b1 and b2 are what the front end uses to predict uptake
from a proposed lane. Plain numpy OLS keeps it dependency-light and easy to read.
Writes coefficients to frontend/data/model.json.
"""
import csv
import json

import numpy as np

from config import DECAY_M, FRONTEND_DATA, PROCESSED, TRUE_BETA_EXPOSURE, TRUE_BETA_EXPOSURE_SHORT


def main():
    with open(PROCESSED / "features.csv") as f:
        rows = list(csv.DictReader(f))
    dE = np.array([float(r["d_exposure"]) for r in rows])
    short = np.array([float(r["short_commute_share"]) for r in rows])
    y = np.array([float(r["d_cycle_share"]) for r in rows])

    X = np.column_stack([np.ones_like(dE), dE, dE * short])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))

    names = ["intercept", "d_exposure", "d_exposure_x_short"]
    print(f"OLS on {len(y)} LSOAs, R^2 = {r2:.3f}")
    for n, b, s in zip(names, beta, se):
        print(f"  {n:<20} {b:+.5f}  (se {s:.5f})")
    print(f"  true values used to generate data: d_exposure {TRUE_BETA_EXPOSURE:+.5f}, "
          f"x_short {TRUE_BETA_EXPOSURE_SHORT:+.5f}")

    model = {
        "description": "d_cycle_share = b_exposure * dE + b_exposure_short * dE * short_commute_share",
        "b_exposure": float(beta[1]),
        "b_exposure_short": float(beta[2]),
        "decay_m": DECAY_M,
        "r2": float(r2),
        "n": len(y),
    }
    FRONTEND_DATA.mkdir(parents=True, exist_ok=True)
    with open(FRONTEND_DATA / "model.json", "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote {FRONTEND_DATA / 'model.json'}")


if __name__ == "__main__":
    main()
