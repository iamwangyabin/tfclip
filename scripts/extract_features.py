#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfclip.data import DATASET_NAMES
from tfclip.runner import ensure_image_features
from tfclip.models import OpenCLIPBackbone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="outputs/features")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbone = OpenCLIPBackbone(args.model_name, args.pretrained, device=args.device)
    for dataset_name in args.datasets:
        dataset, bundle = ensure_image_features(
            dataset_name=dataset_name,
            data_root=args.data_root,
            cache_root=args.cache_root,
            backbone=backbone,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            f"{dataset.name}/{bundle.model_id}: "
            f"train={len(bundle.train.labels)} val={len(bundle.val.labels)} test={len(bundle.test.labels)}"
        )


if __name__ == "__main__":
    main()
