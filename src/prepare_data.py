"""Download HAM10000, resize to 224x224, save as one uint8 .npy array + CSV index.

Run on a Habrok LOGIN node (compute nodes have no internet).

    python prepare_data.py --out-dir /scratch/$USER/ham10000
"""
import argparse
import sys
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 224
EXPECTED_N_IMAGES = 10015
EXPECTED_N_LESIONS = 7470
# Only the class counts the paper states explicitly (section 2.1).
# The other five classes are not hard-asserted here to avoid baking in unverified numbers.
EXPECTED_CLASS_COUNTS = {
    "nv": 6705,
    "df": 115,
}

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def find_metadata_and_image_dirs(dataset_root: Path):
    metadata_path = next(dataset_root.rglob("HAM10000_metadata.csv"))
    image_dirs = sorted({p.parent for p in dataset_root.rglob("*.jpg")})
    return metadata_path, image_dirs


def build_image_index(image_dirs):
    """Map image_id -> file path, first match wins.

    The Kaggle copy ships the images twice, in `HAM10000_images_part_*` and
    `ham10000_images_part_*`. On a case-insensitive filesystem (macOS) these collapse
    to one pair; on Linux all four appear. Dirs arrive sorted, so uppercase is seen
    first and wins deterministically on both platforms. Duplicates are reported.
    """
    index, duplicates = {}, 0
    for d in image_dirs:
        for p in sorted(d.glob("*.jpg")):
            if p.stem in index:
                duplicates += 1
                continue
            index[p.stem] = p
    if duplicates:
        chosen = sorted({p.parent.name for p in index.values()})
        print(f"Note: skipped {duplicates} duplicate image files; reading from {chosen}")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading (or reusing cached) HAM10000 via kagglehub...")
    dataset_root = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
    print(f"Dataset root: {dataset_root}")

    metadata_path, image_dirs = find_metadata_and_image_dirs(dataset_root)
    print(f"Metadata: {metadata_path}")
    print(f"Image dirs: {image_dirs}")

    meta = pd.read_csv(metadata_path)
    image_index = build_image_index(image_dirs)

    missing = [img_id for img_id in meta["image_id"] if img_id not in image_index]
    if missing:
        print(f"ERROR: {len(missing)} image_ids in metadata have no matching file, e.g. {missing[:5]}")
        sys.exit(1)

    n_images = len(meta)
    n_lesions = meta["lesion_id"].nunique()
    class_counts = meta["dx"].value_counts().to_dict()

    print(f"Images: {n_images} (expected {EXPECTED_N_IMAGES})")
    print(f"Unique lesion_id: {n_lesions} (expected {EXPECTED_N_LESIONS})")
    print(f"Class counts: {class_counts}")
    print(f"Expected (paper, section 2.1): {EXPECTED_CLASS_COUNTS}")

    assert n_images == EXPECTED_N_IMAGES, f"expected {EXPECTED_N_IMAGES} images, got {n_images}"
    assert n_lesions == EXPECTED_N_LESIONS, f"expected {EXPECTED_N_LESIONS} lesions, got {n_lesions}"
    for cls, expected_count in EXPECTED_CLASS_COUNTS.items():
        actual = class_counts.get(cls, 0)
        assert actual == expected_count, f"class {cls}: expected {expected_count}, got {actual}"

    print("Verification passed. Resizing images to 224x224...")

    images = np.empty((n_images, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, img_id in enumerate(tqdm(meta["image_id"], desc="resize")):
        img = Image.open(image_index[img_id]).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        images[i] = np.asarray(img)

    npy_path = out_dir / "ham10000_224.npy"
    csv_path = out_dir / "ham10000_index.csv"
    np.save(npy_path, images)
    meta.reset_index(drop=True).to_csv(csv_path, index_label="array_index")

    print(f"Saved images: {npy_path} ({images.nbytes / 1e9:.2f} GB)")
    print(f"Saved index: {csv_path}")


if __name__ == "__main__":
    main()
