#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfclip.data import DATASET_NAMES
from tfclip.methods import METHOD_NAMES
from tfclip.runner import append_result, run_many


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="outputs/features")
    parser.add_argument("--output", default="outputs/results/baselines.jsonl")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_NAMES))
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        for shots in args.shots:
            for seed in args.seeds:
                records = run_many(
                    dataset_name=dataset,
                    methods=args.methods,
                    shots=shots,
                    seed=seed,
                    model_name=args.model_name,
                    pretrained=args.pretrained,
                    data_root=args.data_root,
                    cache_root=args.cache_root,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                )
                for record in records:
                    append_result(args.output, record)
                    print(
                        f"{record['dataset']} {record['method']} "
                        f"{record['shots']}shot seed={record['seed']} "
                        f"{record['model_id']} acc={record['accuracy']:.2f}"
                    )


if __name__ == "__main__":
    main()
