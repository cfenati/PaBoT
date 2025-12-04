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
    # ensure at least 1 sample per split if possible
    if n_train == 0 and n > 0:
        n_train = 1
    if n_val == 0 and n > n_train + 1:
        n_val = 1
    return n_train, n_val


def copy_pair(
    pt: str,
    sl: str,
    mri_path: Path,
    ct_path: Path,
    compact_mapping: Dict[str, str],
    phase: str,
    output_root: Path,
    dry_run: bool,
) -> None:
    compact_id = compact_mapping[pt]
    dst_name = f"{compact_id}_slice_{sl}{mri_path.suffix.lower()}"  # assume CT shares suffix

    phaseA_dir = output_root / f"{phase}A"
    phaseB_dir = output_root / f"{phase}B"
    phaseA_dir.mkdir(parents=True, exist_ok=True)
    phaseB_dir.mkdir(parents=True, exist_ok=True)

    dst_mri = phaseA_dir / dst_name
    dst_ct = phaseB_dir / dst_name

    if dry_run:
        print(f"[{phase}] Would copy MRI {mri_path} -> {dst_mri}")
        print(f"[{phase}] Would copy CT  {ct_path} -> {dst_ct}")
    else:
        shutil.copy2(mri_path, dst_mri)
        shutil.copy2(ct_path, dst_ct)


def main() -> None:
    args = parse_args()

    mri_paths = collect_paths(args.mri_dir)
    ct_paths = collect_paths(args.ct_dir)

    if not mri_paths:
        raise SystemExit(f"No files found in MRI directory: {args.mri_dir}")
    if not ct_paths:
        raise SystemExit(f"No files found in CT directory: {args.ct_dir}")

    print(f"Found {len(mri_paths)} MRI files and {len(ct_paths)} CT files.")

    # 1) Pair MRI/CT by patient + slice
    pairs = pair_mri_ct_slices(mri_paths, ct_paths)
    print(f"Using {len(pairs)} paired MRI/CT slices.")

    # 2) Build patient compact IDs from all paths (MRI+CT)
    combined_paths = mri_paths + ct_paths
    compact_mapping = build_compact_ids(combined_paths, args.prefix)
    print(f"Found {len(compact_mapping)} patients.")

    # 3) Shuffle pairs
    random.seed(args.seed)
    random.shuffle(pairs)

    # 4) Compute split sizes
    n = len(pairs)
    n_train, n_val = compute_split_indices(n, args.train_frac, args.val_frac)
    n_test = n - n_train - n_val

    print(f"Total slices: {n}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # 5) Split
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    # 6) Copy to destination
    for phase, phase_pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        for pt, sl, mri_path, ct_path in phase_pairs:
            copy_pair(
                pt=pt,
                sl=sl,
                mri_path=mri_path,
                ct_path=ct_path,
                compact_mapping=compact_mapping,
                phase=phase,
                output_root=args.output,
                dry_run=args.dry_run,
            )

    # 7) Save mapping
    mapping_path = args.output / "patient_mapping.json"
    if args.dry_run:
        print(f"Would write patient mapping to {mapping_path}")
    else:
        mapping_path.write_text(json.dumps(compact_mapping, indent=2))
        print(f"Saved patient ID mapping to {mapping_path}")


if __name__ == "__main__":
    main()
