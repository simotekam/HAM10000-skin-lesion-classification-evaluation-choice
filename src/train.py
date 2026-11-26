"""One run = (split, config, seed): train, early-stop on val macro-F1, evaluate on test.

Usage:
    python train.py --task-id 0                     # looks up (split, config, seed) in RUN_TABLE
    python train.py --split A --config full --seed 0  # explicit override, e.g. for smoke tests
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from losses import build_loss, class_weights_from_counts

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # alphabetical, fixed label order
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEL_IDX = CLASS_TO_IDX["mel"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tier 1 (must): config=full, split A vs B, seeds 0-4 -> task-id 0-9
# Tier 2: split B, no_fl / no_cw, seeds 0-4 -> task-id 10-19
# Tier 3 (if time): split B, plain, seeds 0-4 -> task-id 20-24
RUN_TABLE = (
    [{"split": s, "config": "full", "seed": k} for s in ["A", "B"] for k in range(5)]
    + [{"split": "B", "config": c, "seed": k} for c in ["no_fl", "no_cw"] for k in range(5)]
    + [{"split": "B", "config": "plain", "seed": k} for k in range(5)]
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HAM10000Dataset(Dataset):
    def __init__(self, images: np.ndarray, df: pd.DataFrame, transform):
        self.images = images
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.fromarray(self.images[row["array_index"]])
        label = CLASS_TO_IDX[row["dx"]]
        return self.transform(img), label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASSES)),
    )
    return model


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train(train)
    total_loss, all_probs, all_labels = 0.0, [], []
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_probs.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(labels)
    macro_f1 = f1_score(labels, probs.argmax(1), average="macro")
    return avg_loss, macro_f1, probs, labels


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=range(len(CLASSES)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))

    mel_true = (labels == MEL_IDX).astype(int)
    mel_pred = (preds == MEL_IDX).astype(int)
    tp = int(((mel_pred == 1) & (mel_true == 1)).sum())
    fn = int(((mel_pred == 0) & (mel_true == 1)).sum())
    tn = int(((mel_pred == 0) & (mel_true == 0)).sum())
    fp = int(((mel_pred == 1) & (mel_true == 0)).sum())
    mel_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    mel_specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    try:
        macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=range(len(CLASSES)))
    except ValueError:
        macro_auc = float("nan")  # e.g. a class absent from this subgroup

    return {
        "n": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_auc_ovr": macro_auc,
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "melanoma_sensitivity": mel_sensitivity,
        "melanoma_specificity": mel_specificity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--split", choices=["A", "B"])
    parser.add_argument("--config", choices=["full", "no_fl", "no_cw", "plain"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--splits-dir", default=str(ROOT / "splits"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7, help="early stop after N epochs with no val macro-F1 improvement")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="smoke-test override: subsample before training")
    args = parser.parse_args()

    if args.task_id is not None:
        run = RUN_TABLE[args.task_id]
        split, config, seed = run["split"], run["config"], run["seed"]
    else:
        split, config, seed = args.split, args.config, args.seed
        if split is None or config is None or seed is None:
            parser.error("pass --task-id, or all of --split/--config/--seed")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"split={split} config={config} seed={seed} device={device}")

    data_dir, splits_dir = Path(args.data_dir), Path(args.splits_dir)
    images = np.load(data_dir / "ham10000_224.npy", mmap_mode="r")
    suffix = split.lower()
    train_df = pd.read_csv(splits_dir / f"train_{suffix}.csv")
    val_df = pd.read_csv(splits_dir / f"val_{suffix}.csv")
    test_df = pd.read_csv(splits_dir / f"test_{suffix}.csv")

    if args.max_images is not None:  # smoke test: shrink without touching the real split files
        def stratified_subsample(df, n):
            per_class = max(1, n // len(CLASSES))
            parts = [g.sample(n=min(per_class, len(g)), random_state=0) for _, g in df.groupby("dx")]
            return pd.concat(parts)
        train_df = stratified_subsample(train_df, args.max_images)
        val_df = stratified_subsample(val_df, args.max_images)
        test_df = stratified_subsample(test_df, args.max_images)

    train_tf, eval_tf = build_transforms()
    train_ds = HAM10000Dataset(images, train_df, train_tf)
    val_ds = HAM10000Dataset(images, val_df, eval_tf)
    test_ds = HAM10000Dataset(images, test_df, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    counts = torch.tensor([(train_df["dx"] == c).sum() for c in CLASSES])
    class_weights = class_weights_from_counts(counts).to(device)
    loss_fn = build_loss(config, class_weights, gamma=2.0).to(device)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    out_dir = Path(args.output_dir) / split / config / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1, best_epoch, epochs_no_improve = -1.0, -1, 0
    best_state = None
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        scheduler.step(val_loss)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, epochs_no_improve = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    test_loss, test_f1, test_probs, test_labels = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False)
    metrics = compute_metrics(test_labels, test_probs)

    result = {
        "split": split,
        "config": config,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "train_seconds": time.time() - t0,
        "test_metrics": metrics,
    }

    if split == "A" and "seen_lesion" in test_df.columns:
        seen_mask = (test_df["seen_lesion"] == "seen").to_numpy()
        result["test_metrics_seen"] = compute_metrics(test_labels[seen_mask], test_probs[seen_mask])
        result["test_metrics_unseen"] = compute_metrics(test_labels[~seen_mask], test_probs[~seen_mask])

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(out_dir / "test_probs.npz", probs=test_probs, labels=test_labels)
    print(f"done. best_epoch={best_epoch} test_acc={metrics['accuracy']:.4f} test_macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
