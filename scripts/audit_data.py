#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfclip.data import DATASET_NAMES, load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.datasets:
        print(f"auditing {name}...", flush=True)
        dataset = load_dataset(name, args.data_root, seed=args.seed)
        missing = []
        for split_name in ("train", "val", "test"):
            for item in getattr(dataset, split_name):
                if not Path(item.path).exists():
                    missing.append(str(item.path))
        print(
            f"{dataset.name}: classes={len(dataset.classnames)} "
            f"train={len(dataset.train)} val={len(dataset.val)} test={len(dataset.test)} "
            f"missing={len(missing)}",
            flush=True,
        )
        if missing:
            for path in missing[:10]:
                print(f"  missing: {path}")


if __name__ == "__main__":
    main()
