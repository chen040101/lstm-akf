from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_json_dataset(
    source_dir: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    source_dir = Path(source_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.glob("*.json"))
    random.Random(seed).shuffle(files)
    val_count = int(len(files) * val_ratio)
    val_files = set(files[:val_count])

    for file_path in files:
        destination = val_dir if file_path in val_files else train_dir
        if file_path.parent.resolve() == destination.resolve():
            continue
        target = destination / file_path.name
        if target.exists():
            target.unlink()
        shutil.move(str(file_path), str(target))

    return len(files) - val_count, val_count


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Split JSON samples into train and validation sets.")
    parser.add_argument("--source-dir", type=Path, default=root / "data" / "dataset" / "train")
    parser.add_argument("--train-dir", type=Path, default=root / "data" / "dataset" / "train")
    parser.add_argument("--val-dir", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_json_dataset(
        source_dir=args.source_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


__all__ = ["main", "parse_args", "split_json_dataset"]
