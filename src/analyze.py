"""Aggregate results.json files into tables, Welch t-tests and figures.

Run after the SLURM arrays finish.

Usage:
    python analyze.py --results-dir /scratch/$USER/ham_results --out-dir report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HEADLINE_METRICS = [
    "accuracy", "macro_f1", "balanced_accuracy", "macro_auc_ovr",
    "melanoma_sensitivity", "melanoma_specificity",
]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.rglob("results.json")):
        with open(path) as f:
            r = json.load(f)
        row = {
            "split": r["split"], "config": r["config"], "seed": r["seed"],
            "best_epoch": r["best_epoch"], "best_val_macro_f1": r["best_val_macro_f1"],
        }
        for m in HEADLINE_METRICS:
            row[m] = r["test_metrics"][m]
        for group in ["seen", "unseen"]:
            key = f"test_metrics_{group}"
            if key in r:
                for m in HEADLINE_METRICS:
                    row[f"{group}_{m}"] = r[key][m]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No results.json found under {results_dir}")
    return pd.DataFrame(rows)


def mean_sem_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = []
    for (split, config), g in df.groupby(["split", "config"]):
        row = {"split": split, "config": config, "n_seeds": len(g)}
        for m in metrics:
            if m not in g or g[m].isna().all():
                continue
            vals = g[m].dropna()
            row[m] = f"{vals.mean():.4f} ± {stats.sem(vals):.4f}" if len(vals) > 1 else f"{vals.mean():.4f}"
        out.append(row)
    return pd.DataFrame(out)


def welch(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "p": float("nan"), "cohens_d": float("nan")}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
    return {"t": t, "p": p, "cohens_d": d}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", default="report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.results_dir))
    df.to_csv(out_dir / "all_runs.csv", index=False)
    print(f"Loaded {len(df)} runs: "
          + ", ".join(f"{s}/{c}×{n}" for (s, c), n in df.groupby(['split', 'config']).size().items()))

    print("\n" + "=" * 78)
    print("Mean ± SEM over seeds (test set)")
    print("=" * 78)
    table = mean_sem_table(df, HEADLINE_METRICS)
    print(table.to_string(index=False))
    table.to_csv(out_dir / "mean_sem.csv", index=False)

    # RQ1: split A vs split B, config full
    a = df[(df["split"] == "A") & (df["config"] == "full")]
    b = df[(df["split"] == "B") & (df["config"] == "full")]
    if len(a) and len(b):
        print("\n" + "=" * 78)
        print("RQ1: Split A (image-level) vs Split B (lesion-level), config=full")
        print("=" * 78)
        rows = []
        for m in HEADLINE_METRICS:
            r = welch(a[m], b[m])
            rows.append({"metric": m, "split_A": a[m].mean(), "split_B": b[m].mean(),
                         "diff_A_minus_B": a[m].mean() - b[m].mean(), **r})
        rq1 = pd.DataFrame(rows)
        print(rq1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        rq1.to_csv(out_dir / "rq1_split_comparison.csv", index=False)

    # RQ1 (cleanest evidence): seen vs unseen lesions inside split A
    seen_cols = [f"seen_{m}" for m in HEADLINE_METRICS if f"seen_{m}" in df.columns]
    if seen_cols and len(a):
        print("\n" + "=" * 78)
        print("RQ1b: within Split A, test images whose lesion is seen vs unseen in train")
        print("=" * 78)
        rows = []
        for m in HEADLINE_METRICS:
            if f"seen_{m}" not in a.columns:
                continue
            r = welch(a[f"seen_{m}"], a[f"unseen_{m}"])
            rows.append({"metric": m, "seen": a[f"seen_{m}"].mean(), "unseen": a[f"unseen_{m}"].mean(),
                         "diff_seen_minus_unseen": a[f"seen_{m}"].mean() - a[f"unseen_{m}"].mean(), **r})
        rq1b = pd.DataFrame(rows)
        print(rq1b.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        rq1b.to_csv(out_dir / "rq1b_seen_vs_unseen.csv", index=False)

    # RQ2: each config vs full, under split B
    full_b = df[(df["split"] == "B") & (df["config"] == "full")]
    others = [c for c in df[df["split"] == "B"]["config"].unique() if c != "full"]
    if len(full_b) and others:
        print("\n" + "=" * 78)
        print("RQ2: ablations vs config=full, under Split B (lesion-level)")
        print("=" * 78)
        rows = []
        for config in sorted(others):
            g = df[(df["split"] == "B") & (df["config"] == config)]
            for m in ["accuracy", "macro_f1", "balanced_accuracy", "melanoma_sensitivity"]:
                r = welch(g[m], full_b[m])
                rows.append({"config": config, "metric": m, "config_mean": g[m].mean(),
                             "full_mean": full_b[m].mean(),
                             "diff_vs_full": g[m].mean() - full_b[m].mean(), **r})
        rq2 = pd.DataFrame(rows)
        print(rq2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        rq2.to_csv(out_dir / "rq2_ablations.csv", index=False)

    print(f"\nWrote CSVs to {out_dir}/")
    print("Note: 5 seeds is a small sample; read effect sizes (cohens_d) alongside p-values.")
    print("Significance refers to variation across training runs on a fixed test set.")


if __name__ == "__main__":
    main()
