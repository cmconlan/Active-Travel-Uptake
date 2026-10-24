"""Generate SYNTHETIC 2011 and 2021 cycle-to-work counts for each Brighton and Hove LSOA.

The LSOAs, network and features are real (OSM); the census numbers are not. They are generated
from the real features using known effects (TRUE_EFFECTS_PP in config.py), so train_model.py can
show that the pipeline recovers them. Nothing here describes real travel behaviour.

  workers      synthetic working residents (~550-1,000 per LSOA)
  share_2011   ~4% on average, higher where 2015 routes were calmer and closer to the centre
  change       planted effects from model_spec.design() + LSOA noise
  counts       Binomial(workers, share), so small-number sampling noise is realistic too

Outputs: data/processed/census_synthetic.csv   lsoa21cd, year, workers, cycle_to_work
         data/processed/synthetic_truth.json   the planted effects and the scaler they used
"""
import json

import numpy as np
import pandas as pd

from config import (CENSUS_YEARS, NOISE_SD_PP, PROCESSED, SNAPSHOTS, SYNTHETIC_SEED,
                    TRUE_EFFECTS_PP)
from model_spec import CHANGE_TERMS, CONTROL_TERMS, INTERACTION, TERMS, design, fit_scaler, training_frame

DISCLAIMER = ("SYNTHETIC DATA: generated for a software demo from planted effects. "
              "Does not represent real travel behaviour in Brighton and Hove.")


def main():
    rng = np.random.default_rng(SYNTHETIC_SEED)
    y_net0 = min(SNAPSHOTS)
    c0, c1 = CENSUS_YEARS
    feats = pd.read_csv(PROCESSED / "lsoa_features.csv")
    df = training_frame(feats, y_net0)
    needed = [*CHANGE_TERMS, *CONTROL_TERMS, INTERACTION[1], f"mean_route_lts_{y_net0}"]
    usable = df.dropna(subset=needed)
    if len(usable) < len(df):
        print(f"  {len(df) - len(usable)} LSOAs without complete features are left out")
    df = usable.reset_index(drop=True)

    scaler = fit_scaler(df)
    X = design(df, scaler)
    beta = np.array([TRUE_EFFECTS_PP[t] for t in TERMS])
    change_pp = X.to_numpy() @ beta + rng.normal(0, NOISE_SD_PP, len(df))

    z = lambda s: (s - s.mean()) / s.std()
    share0 = (0.04 - 0.008 * z(df[f"mean_route_lts_{y_net0}"]) + 0.006 * z(df[f"ls_reachable_pois_{y_net0}"])
              - 0.008 * z(df["dist_centre_m"]) + rng.normal(0, 0.008, len(df))).clip(0.005, 0.3)
    share1 = (share0 + change_pp / 100).clip(0.003, 0.4)
    workers0 = rng.integers(550, 1000, len(df))
    workers1 = (workers0 * rng.uniform(0.95, 1.08, len(df))).astype(int)

    rows = []
    for year, workers, share in ((c0, workers0, share0), (c1, workers1, share1)):
        cyclists = rng.binomial(workers, share)
        rows.append(pd.DataFrame({"lsoa21cd": df["lsoa21cd"], "year": year,
                                  "workers": workers, "cycle_to_work": cyclists}))
    census = pd.concat(rows, ignore_index=True).sort_values(["lsoa21cd", "year"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    census.to_csv(PROCESSED / "census_synthetic.csv", index=False)
    with open(PROCESSED / "synthetic_truth.json", "w") as f:
        json.dump({"disclaimer": DISCLAIMER, "effects_pp": TRUE_EFFECTS_PP, "noise_sd_pp": NOISE_SD_PP,
                   "seed": SYNTHETIC_SEED, "scaler": scaler}, f, indent=2)

    wide = census.pivot(index="lsoa21cd", columns="year")
    s0 = wide[("cycle_to_work", c0)] / wide[("workers", c0)]
    s1 = wide[("cycle_to_work", c1)] / wide[("workers", c1)]
    print(f"{DISCLAIMER}\n")
    print(f"Wrote census_synthetic.csv for {len(df)} LSOAs: cycle-to-work share {s0.mean():.1%} ({c0}) -> "
          f"{s1.mean():.1%} ({c1}); change by LSOA ranges {100 * (s1 - s0).min():+.1f} to "
          f"{100 * (s1 - s0).max():+.1f} pp")


if __name__ == "__main__":
    main()
