#!/usr/bin/env python3
"""Recompute Study 1 summary statistics and split-half reliability from public ratings."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SEED = 20260827
N_REPETITIONS = 1000
DIMENSIONS = ["vision", "audition", "gustation", "olfaction", "touch", "interoception",
              "leg_foot", "hand_arm", "head", "mouth_throat", "torso"]

def split_half(data):
    rng = np.random.default_rng(SEED)
    rows = []
    for dimension in DIMENSIONS:
        arrays = [g["rating"].to_numpy(float) for _, g in
                  data.loc[data["dimension"] == dimension].groupby("classifier_id", sort=True)]
        corrected = []
        for _ in range(N_REPETITIONS):
            a, b = [], []
            for values in arrays:
                values = rng.permutation(values)
                cut = (len(values) + 1) // 2
                a.append(values[:cut].mean())
                b.append(values[cut:].mean())
            r = float(np.corrcoef(a, b)[0, 1])
            corrected.append(2 * r / (1 + r))
        x = np.asarray(corrected)
        rows.append({"dimension": dimension, "seed": SEED, "n_repetitions": N_REPETITIONS,
                     "spearman_brown_mean": x.mean(), "spearman_brown_median": np.median(x),
                     "spearman_brown_ci95_lower": np.quantile(x, .025),
                     "spearman_brown_ci95_upper": np.quantile(x, .975)})
    return pd.DataFrame(rows)

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=root / "Study1_anonymized_ratings.csv")
    parser.add_argument("--published", type=Path, default=root / "Classifier_sensorimotor_norms.csv")
    parser.add_argument("--output", type=Path, default=root / "reproduced_outputs" / "study1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(args.ratings)
    grouped = ratings.groupby(["classifier_id", "classifier", "classifier_hanzi", "dimension"], sort=True)["rating"]
    long = grouped.agg(mean="mean", sd="std", n="count").reset_index()
    long.to_csv(args.output / "reproduced_classifier_norms_long.csv", index=False, encoding="utf-8-sig")
    split_half(ratings).to_csv(args.output / "reproduced_split_half_reliability.csv", index=False, encoding="utf-8-sig")

    published = pd.read_csv(args.published).set_index("classifier_id")
    checks = []
    for dim in DIMENSIONS:
        observed = long.loc[long.dimension == dim].set_index("classifier_id")
        for stat in ["mean", "sd", "n"]:
            delta = observed[stat].astype(float) - published.loc[observed.index, f"{dim}_{stat}"].astype(float)
            checks.append({"dimension": dim, "statistic": stat, "max_abs_difference": float(delta.abs().max())})
    pd.DataFrame(checks).to_csv(args.output / "norm_reproduction_check.csv", index=False, encoding="utf-8-sig")
    print(f"Outputs written to {args.output}")

if __name__ == "__main__":
    main()
