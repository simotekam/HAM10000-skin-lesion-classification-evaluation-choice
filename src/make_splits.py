"""Build Split A (image-level) and Split B (lesion-level) over the preprocessed index.

Seed fixed at 0. Writes CSVs to splits/.

    python make_splits.py --data-dir /scratch/$USER/ham10000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED = 0
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "splits"


def split_a_image_level(meta: pd.DataFrame, seed: int):
    """Random stratified split over all images, ignoring lesion_id."""
    train_val_idx, test_idx = train_test_split(
        meta.index, test_size=0.2, stratify=meta["dx"], random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        stratify=meta.loc[train_val_idx, "dx"],
        random_state=seed,
    )
    return (
        meta.loc[train_idx].copy(),
        meta.loc[val_idx].copy(),
        meta.loc[test_idx].copy(),
    )


def split_b_lesion_level(meta: pd.DataFrame, seed: int):
    """Stratified group split by lesion_id: one fold out as test, then again for val."""
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(
        sgkf_outer.split(meta, meta["dx"], groups=meta["lesion_id"])
    )
    train_val = meta.iloc[train_val_idx]

    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(
        sgkf_inner.split(train_val, train_val["dx"], groups=train_val["lesion_id"])
    )
    train = train_val.iloc[train_sub_idx].copy()
    val = train_val.iloc[val_sub_idx].copy()
    test = meta.iloc[test_idx].copy()
    return train, val, test


def assert_no_lesion_overlap(train, val, test):
    ids_train = set(train["lesion_id"])
    ids_val = set(val["lesion_id"])
    ids_test = set(test["lesion_id"])
    assert not (ids_train & ids_val), f"train/val lesion overlap: {ids_train & ids_val}"
    assert not (ids_train & ids_test), f"train/test lesion overlap: {ids_train & ids_test}"
    assert not (ids_val & ids_test), f"val/test lesion overlap: {ids_val & ids_test}"


def tag_seen_unseen(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """For split A: does another image of this test lesion appear in train?"""
    train_lesions = set(train["lesion_id"])
    return test["lesion_id"].isin(train_lesions).map({True: "seen", False: "unseen"})


def subset_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    counts = df["dx"].value_counts()
    row = {"subset": name, "n_images": len(df), "n_lesions": df["lesion_id"].nunique()}
    row.update(counts.to_dict())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(args.data_dir) / "ham10000_index.csv")

    # sanity check assumed in split_b_lesion_level: dx is constant per lesion_id
    dx_per_lesion = meta.groupby("lesion_id")["dx"].nunique()
    assert (dx_per_lesion == 1).all(), "found a lesion_id with more than one dx label"

    print("=" * 70)
    print("SPLIT A: image-level (random stratified by dx)")
    print("=" * 70)
    train_a, val_a, test_a = split_a_image_level(meta, SEED)
    test_a = test_a.copy()
    test_a["seen_lesion"] = tag_seen_unseen(train_a, test_a)

    rows_a = [subset_table("train", train_a), subset_table("val", val_a), subset_table("test", test_a)]
    table_a = pd.DataFrame(rows_a).fillna(0)
    print(table_a.to_string(index=False))
    print()
    print("Split A test set, seen vs unseen lesion:")
    print(test_a["seen_lesion"].value_counts().to_string())

    print()
    print("=" * 70)
    print("SPLIT B: lesion-level (StratifiedGroupKFold by lesion_id)")
    print("=" * 70)
    train_b, val_b, test_b = split_b_lesion_level(meta, SEED)
    assert_no_lesion_overlap(train_b, val_b, test_b)
    print("Overlap assertion passed: zero lesion_id overlap between train/val/test.")

    rows_b = [subset_table("train", train_b), subset_table("val", val_b), subset_table("test", test_b)]
    table_b = pd.DataFrame(rows_b).fillna(0)
    print(table_b.to_string(index=False))

    for name, df in [
        ("train_a", train_a), ("val_a", val_a), ("test_a", test_a),
        ("train_b", train_b), ("val_b", val_b), ("test_b", test_b),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
    print()
    print(f"Saved 6 split CSVs to {out_dir}")


if __name__ == "__main__":
    main()
