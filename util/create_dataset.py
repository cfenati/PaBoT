"""Utility to organize MRI/CT PNG files into the PaBoT folder layout with
random train/val/test splits at the slice level.

Example:
    python util/organize_pngs.py \
        --mri-dir /path/to/mri_images \
        --ct-dir /path/to/ct_images \
        --output ./datasets/MyStudy \
        --train-frac 0.7 --val-frac 0.15 --seed 42

This will create trainA/trainB, valA/valB, testA/testB under --output,
with filenames like p001_slice_0001.png and a patient_mapping.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize MRI/CT PNGs into PaBoT layout with random train/val/test splits.",
    )
    parser.add_argument("--mri-dir", type=Path, required=True, help="Directory containing MRI PNG slices.")
    parser.add_argument("--ct-dir", type=Path, required=True, help="Directory containing CT PNG slices.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory where trainA/trainB/valA/... will be created (e.g., datasets/MyStudy).",
    )
    parser.add_argument(
        "--prefix",
        default="p",
        help="Prefix for compact patient IDs (default: 'p' -> p001, p002, ...).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of slices for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of slices for validation (default: 0.15). Test gets the rest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling slices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned copies without writing files.",
    )
    return parser.parse_args()


def infer_patient_token(path: Path) -> str:
    # Adapt this if your naming scheme is different
    stem = path.stem
    return stem.split("_")[0]


def infer_slice_number(path: Path) -> str:
    # Looks for "slice123" or "slice_123" or "slice-123" in the filename
    match = re.search(r"slice[-_]?([0-9]+)", path.stem, re.IGNORECASE)
    if match:
        return match.group(1).zfill(4)
    # fallback if no slice pattern: treat whole stem as number
    digits = re.findall(r"(\d+)", path.stem)
    if digits:
        return digits[-1].zfill(4)
    return "0000"


def build_compact_ids(paths: Iterable[Path], prefix: str) -> Dict[str, str]:
    tokens = sorted({infer_patient_token(p) for p in paths})
    return {token: f"{prefix}{idx:03d}" for idx, token in enumerate(tokens, start=1)}


def collect_paths(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def pair_mri_ct_slices(
    mri_paths: List[Path],
    ct_paths: List[Path],
) -> List[Tuple[str, str, Path, Path]]:
    """Return list of (patient_token, slice_id, mri_path, ct_path)."""
    ct_index: Dict[Tuple[str, str], Path] = {}
    for ct in ct_paths:
        pt = infer_patient_token(ct)
        sl = infer_slice_number(ct)
        key = (pt, sl)
        if key in ct_index:
            raise RuntimeError(f"Duplicate CT slice key {key} for files {ct_index[key]} and {ct}")
        ct_index[key] = ct

    pairs: List[Tuple[str, str, Path, Path]] = []
    missing_ct = 0

    for mri in mri_paths:
        pt = infer_patient_token(mri)
        sl = infer_slice_number(mri)
        key = (pt, sl)
        ct = ct_index.get(key)
        if ct is None:
            missing_ct += 1
            print(f"WARNING: No CT match for MRI slice {mri} (key={key})")
            continue
        pairs.append((pt, sl, mri, ct))

    if missing_ct:
        print(f"WARNING: {missing_ct} MRI slices had no matching CT slice.")

    if not pairs:
        raise SystemExit("No MRI/CT slice pairs found. Check naming and directories.")

    return pairs


def compute_split_indices(
    n: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[int, int]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise ValueError("Require train_frac > 0, val_frac >= 0, and train_frac + val_frac < 1.")
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # try to ensure at least 1 sample per split when possible
    if n_train == 0 and n > 0:
        n_train = 1
    if n_val == 0 and n > n_train + 1:
        n_val = 1
    return n_train, n_val


def split_paths(
    paths: List[Path],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Randomly split a list of paths into train/val/test (UNPAIRED)."""
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n = len(paths)
    n_train, n_val = compute_split_indices(n, train_frac, val_frac)
    n_test = n - n_train - n_val

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


def copy_group(
    paths: Iterable[Path],
    phase: str,  # "train" / "val" / "test"
    is_mri: bool,
    compact_mapping: Dict[str, str],
    output_root: Path,
    dry_run: bool,
) -> None:
    """Copy ONE modality (MRI or CT) into phaseA or phaseB."""
    phase_dir = output_root / f"{phase}{'A' if is_mri else 'B'}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    for src in paths:
        patient_token = infer_patient_token(src)
        compact_id = compact_mapping[patient_token]
        slice_id = infer_slice_number(src)
        dst_name = f"{compact_id}_slice_{slice_id}{src.suffix.lower()}"
        dst = phase_dir / dst_name

        if dry_run:
            print(f"[{phase}][{'MRI' if is_mri else 'CT '}] Would copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    mri_paths = collect_paths(args.mri_dir)
    ct_paths = collect_paths(args.ct_dir)

    if not mri_paths:
        raise SystemExit(f"No files found in MRI directory: {args.mri_dir}")
    if not ct_paths:
        raise SystemExit(f"No files found in CT directory: {args.ct_dir}")

    print(f"Found {len(mri_paths)} MRI files and {len(ct_paths)} CT files.")

    # Build patient compact IDs from all files (MRI + CT)
    combined_paths = mri_paths + ct_paths
    compact_mapping = build_compact_ids(combined_paths, args.prefix)
    print(f"Found {len(compact_mapping)} unique patients.")

    # Randomly split MRI and CT slices independently (UNPAIRED)
    mri_train, mri_val, mri_test = split_paths(
        mri_paths, args.train_frac, args.val_frac, seed=args.seed
    )
    ct_train, ct_val, ct_test = split_paths(
        ct_paths, args.train_frac, args.val_frac, seed=args.seed + 1  # different seed just to be safe
    )

    print("MRI split:  train =", len(mri_train), "val =", len(mri_val), "test =", len(mri_test))
    print("CT split:   train =", len(ct_train), "val =", len(ct_val), "test =", len(ct_test))

    # Copy MRI slices to *A folders, CT slices to *B folders
    for phase, mri_group, ct_group in [
        ("train", mri_train, ct_train),
        ("val",   mri_val,  ct_val),
        ("test",  mri_test, ct_test),
    ]:
        copy_group(mri_group, phase=phase, is_mri=True,  compact_mapping=compact_mapping,
                   output_root=args.output, dry_run=args.dry_run)
        copy_group(ct_group,  phase=phase, is_mri=False, compact_mapping=compact_mapping,
                   output_root=args.output, dry_run=args.dry_run)

    # Save mapping
    mapping_path = args.output / "patient_mapping.json"
    if args.dry_run:
        print(f"Would write patient mapping to {mapping_path}")
    else:
        mapping_path.write_text(json.dumps(compact_mapping, indent=2))
        print(f"Saved patient ID mapping to {mapping_path}")


if __name__ == "__main__":
    main()
