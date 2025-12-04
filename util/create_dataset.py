"""Utility to organize MRI/CT PNG files into the PaBoT folder layout with compact, consistent filenames.

Example:
    python util/organize_pngs.py \
        --mri-dir /path/to/mri_images \
        --ct-dir /path/to/ct_images \
        --output ./datasets/MyStudy

The script will create trainA/trainB folders under --output and copy images
with names like p001_slice_0001.png while preserving slice indices.
It also writes patient_mapping.json so you can trace original IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize MRI/CT PNGs into PaBoT trainA/trainB folders with compact filenames.",
    )
    parser.add_argument("--mri-dir", type=Path, required=True, help="Directory containing MRI PNG slices.")
    parser.add_argument("--ct-dir", type=Path, required=True, help="Directory containing CT PNG slices.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory where trainA/trainB will be created (e.g., datasets/MyStudy).",
    )
    parser.add_argument(
        "--phase",
        default="train",
        help="Dataset split name to create (train/val/test). Default: train.",
    )
    parser.add_argument(
        "--prefix",
        default="p",
        help="Prefix for compact patient IDs (default: 'p' -> p001, p002, ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without copying files.",
    )
    return parser.parse_args()


def infer_patient_token(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0]


def infer_slice_number(path: Path) -> str:
    match = re.search(r"slice[-_]?([0-9]+)", path.stem, re.IGNORECASE)
    if match:
        return match.group(1).zfill(4)
    return "0000"


def build_compact_ids(paths: Iterable[Path], prefix: str) -> Dict[str, str]:
    tokens = sorted({infer_patient_token(p) for p in paths})
    return {token: f"{prefix}{idx:03d}" for idx, token in enumerate(tokens, start=1)}


def collect_paths(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def plan_transfers(
    paths: Iterable[Path],
    mapping: Dict[str, str],
    phase_dir: Path,
) -> List[Tuple[Path, Path]]:
    transfers: List[Tuple[Path, Path]] = []
    for src in paths:
        patient_token = infer_patient_token(src)
        compact_id = mapping[patient_token]
        slice_id = infer_slice_number(src)
        dst_name = f"{compact_id}_slice_{slice_id}{src.suffix.lower()}"
        transfers.append((src, phase_dir / dst_name))
    return transfers


def copy_transfers(transfers: Iterable[Tuple[Path, Path]], dry_run: bool) -> None:
    for src, dst in transfers:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"Would copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)


def write_mapping(mapping: Dict[str, str], output_root: Path, dry_run: bool) -> None:
    mapping_path = output_root / "patient_mapping.json"
    if dry_run:
        print(f"Would write mapping to {mapping_path}")
        return
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print(f"Saved patient ID mapping to {mapping_path}")


def main() -> None:
    args = parse_args()

    mri_paths = collect_paths(args.mri_dir)
    ct_paths = collect_paths(args.ct_dir)

    if not mri_paths:
        raise SystemExit(f"No files found in MRI directory: {args.mri_dir}")
    if not ct_paths:
        raise SystemExit(f"No files found in CT directory: {args.ct_dir}")

    combined_paths = mri_paths + ct_paths
    mapping = build_compact_ids(combined_paths, args.prefix)

    phaseA_dir = args.output / f"{args.phase}A"
    phaseB_dir = args.output / f"{args.phase}B"

    transfers = plan_transfers(mri_paths, mapping, phaseA_dir)
    transfers += plan_transfers(ct_paths, mapping, phaseB_dir)

    copy_transfers(transfers, args.dry_run)
    write_mapping(mapping, args.output, args.dry_run)


if __name__ == "__main__":
    main()