"""Fit the first-difference model of cycle-to-work uptake on network stress change.

  y = 100 * (cycle share 2021 - cycle share 2011), in percentage points, per LSOA
  X = model_spec.design(...): network change terms, reach x baseline interaction, level controls

OLS with HC1 (heteroskedasticity-robust) standard errors, plus k-fold cross-validated R^2.
Because the census is synthetic, the planted effects are known: the script reports whether
each 95% confidence interval covers its true value.

Outputs: data/processed/model.json        coefficients + scaler (what the front end needs)
         data/processed/model_report.csv  estimate, SE, CI, truth per term
         data/processed/model_fit.png     coefficient recovery + observed vs fitted
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CENSUS_YEARS, CV_FOLDS, PROCESSED, SNAPSHOTS, SYNTHETIC_SEED
from model_spec import (CHANGE_TERMS, CONTROL_TERMS, INTERACTION, INTERACTION_TERM, TERMS, design,
                        fit_scaler, training_frame)


def ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ (X * resid[:, None] ** 2)
    se = np.sqrt(np.diag(bread @ meat @ bread) * n / (n - k))  # HC1
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


def cv_r2(df: pd.DataFrame, y: np.ndarray, folds: int, seed: int) -> float:
    """Out-of-sample R^2: scaler and coefficients refit on each training fold."""
    idx = np.random.default_rng(seed).permutation(len(df))
    pred = np.empty(len(df))
    for test in np.array_split(idx, folds):
        train = np.setdiff1d(idx, test)
        sc = fit_scaler(df.iloc[train])
        b, *_ = ols(design(df.iloc[train], sc).to_numpy(), y[train])
        pred[test] = design(df.iloc[test], sc).to_numpy() @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def load():
    c0, c1 = CENSUS_YEARS
    census = pd.read_csv(PROCESSED / "census_synthetic.csv")
    wide = census.pivot(index="lsoa21cd", columns="year", values=["workers", "cycle_to_work"])
    share = lambda y: wide[("cycle_to_work", y)] / wide[("workers", y)]
    outcome = (100 * (share(c1) - share(c0))).rename("d_share_pp").reset_index()
    feats = training_frame(pd.read_csv(PROCESSED / "lsoa_features.csv"), min(SNAPSHOTS))
    df = feats.merge(outcome, on="lsoa21cd").dropna(subset=[*CHANGE_TERMS, *CONTROL_TERMS, INTERACTION[1]])
    return df.reset_index(drop=True)


def plot(report: pd.DataFrame, y, fitted, r2, r2_cv) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    rep = report.iloc[::-1]
    pos = np.arange(len(rep))
    a.errorbar(rep["estimate"], pos, xerr=1.96 * rep["se"], fmt="o", color="#1f5f86", capsize=3, label="estimate, 95% CI")
    if rep["truth"].notna().any():
        a.scatter(rep["truth"], pos, marker="x", s=70, color="#e0572b", zorder=3, label="planted truth (synthetic)")
    a.axvline(0, color="#999999", lw=0.8)
    a.set_yticks(pos, rep.index)
    a.set_xlabel("effect on change in cycle-to-work share (pp per SD)")
    a.legend(frameon=False, loc="lower right")
    a.set_title("Coefficient recovery")

    b.scatter(fitted, y, s=14, alpha=0.7, color="#1f5f86")
    lims = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    b.plot(lims, lims, color="#999999", lw=0.8)
    b.set_xlabel("fitted change (pp)"); b.set_ylabel("observed change (pp)")
    b.set_title(f"Observed vs fitted   R² {r2:.2f}, {CV_FOLDS}-fold CV R² {r2_cv:.2f}")
    fig.suptitle("SYNTHETIC census data: a demonstration of the pipeline, not a finding", color="#b2182b")
    fig.tight_layout()
    fig.savefig(PROCESSED / "model_fit.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    y = df["d_share_pp"].to_numpy()
    scaler = fit_scaler(df)
    X = design(df, scaler)
    beta, se, r2 = ols(X.to_numpy(), y)
    r2_cv = cv_r2(df, y, CV_FOLDS, SYNTHETIC_SEED)

    truth_path = PROCESSED / "synthetic_truth.json"
    truth = json.loads(truth_path.read_text())["effects_pp"] if truth_path.exists() else {}
    report = pd.DataFrame({"estimate": beta, "se": se}, index=TERMS)
    report["ci_low"], report["ci_high"] = beta - 1.96 * se, beta + 1.96 * se
    report["truth"] = [truth.get(t, np.nan) for t in TERMS]
    report["covered"] = (report["truth"] >= report["ci_low"]) & (report["truth"] <= report["ci_high"])
    report.round(4).to_csv(PROCESSED / "model_report.csv", index_label="term")

    model = {
        "disclaimer": "Fitted to SYNTHETIC census data for a software demo. Not a real-world estimate.",
        "outcome": "change in cycle-to-work share, percentage points",
        "terms": TERMS,
        "coefficients": dict(zip(TERMS, map(float, beta))),
        "scaler": scaler,
        "change_terms": CHANGE_TERMS,
        "interaction": {"term": INTERACTION_TERM, "change": INTERACTION[0], "base": INTERACTION[1]},
        "control_terms": CONTROL_TERMS,
        "r2": float(r2), "cv_r2": float(r2_cv), "n": int(len(df)),
    }
    (PROCESSED / "model.json").write_text(json.dumps(model, indent=2))
    plot(report, y, X.to_numpy() @ beta, r2, r2_cv)

    print("SYNTHETIC census: estimates should land near the planted truth.\n")
    print(f"n = {len(df)} LSOAs, R² = {r2:.3f}, {CV_FOLDS}-fold CV R² = {r2_cv:.3f}")
    print(report.round(3).to_string())
    if report["truth"].notna().any():
        print(f"\n{report['covered'].sum()}/{len(report)} 95% intervals cover the planted value "
              f"(expect ~{0.95 * len(report):.0f})")
    print("\nWrote model.json, model_report.csv, model_fit.png")


if __name__ == "__main__":
    main()
