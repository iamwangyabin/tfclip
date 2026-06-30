#!/usr/bin/env python3
"""Prepare StanfordCars from Hugging Face parquet shards.

The original Stanford server is often unavailable. This script converts the
public `tanganke/stanford_cars` parquet shards into the directory structure used
by the CLIP adaptation baselines vendored in this repository.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.io import savemat


TRAIN_FILES = ("train-00000.parquet", "train-00001.parquet")
TEST_FILES = ("test-00000.parquet", "test-00001.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/stanford_cars"),
        help="Output StanfordCars directory.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/stanford_cars/_downloads/tanganke"),
        help="Directory containing tanganke/stanford_cars parquet shards.",
    )
    parser.add_argument(
        "--clip-prompts",
        type=Path,
        default=Path("external/baselines/clip/data/prompts.md"),
        help="CLIP prompt file used to recover StanfordCars class order.",
    )
    parser.add_argument(
        "--class-json",
        type=Path,
        default=Path("external/baselines/ape/gpt3_prompts/CuPL_prompts_stanfordcars.json"),
        help="Fallback JSON whose keys are StanfordCars class names in order.",
    )
    parser.add_argument(
        "--official-split",
        type=Path,
        default=Path("data/stanford_cars/_downloads/split_zhou_StanfordCars.gdrive.html"),
        help="Downloaded Zhou split JSON, used only for train/val class counts.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_class_names(prompt_path: Path, class_json_path: Path) -> list[str]:
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8")
        match = re.search(
            r"## StanfordCars\s+```bash\s+classes = (\[.*?\])\s+templates =",
            text,
            flags=re.DOTALL,
        )
        if match:
            classes = ast.literal_eval(match.group(1))
        elif class_json_path.exists():
            classes = list(json.loads(class_json_path.read_text(encoding="utf-8")).keys())
        else:
            raise RuntimeError(f"Could not parse StanfordCars classes from {prompt_path}")
    elif class_json_path.exists():
        classes = list(json.loads(class_json_path.read_text(encoding="utf-8")).keys())
    else:
        raise FileNotFoundError(f"Missing class sources: {prompt_path} and {class_json_path}")

    if len(classes) != 196:
        raise RuntimeError(f"Expected 196 classes, found {len(classes)}")
    return classes


def year_first(class_name: str) -> str:
    if re.match(r"^\d{4}\b", class_name):
        return class_name
    match = re.match(r"^(.*)\s+(\d{4})$", class_name)
    if not match:
        return class_name
    return f"{match.group(2)} {match.group(1)}"


def official_val_counts(path: Path) -> Counter[int]:
    if not path.exists():
        return Counter()
    split = json.loads(path.read_text(encoding="utf-8"))
    return Counter(int(label) for _, label, _ in split["val"])


def write_image_split(
    parquet_paths: list[Path],
    image_dir: Path,
    image_prefix: str,
    batch_size: int,
) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    items = []
    index = 1

    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels):
                filename = f"{image_prefix}_{index:06d}.jpg"
                output_path = image_dir / filename
                output_path.write_bytes(image["bytes"])
                items.append(
                    {
                        "relative_path": f"{image_dir.name}/{filename}",
                        "filename": filename,
                        "label": int(label),
                    }
                )
                index += 1

    return items


def build_split(
    train_items: list[dict],
    test_items: list[dict],
    class_names: list[str],
    val_counts: Counter[int],
    seed: int,
) -> dict[str, list[list[object]]]:
    by_label = defaultdict(list)
    for item in train_items:
        by_label[item["label"]].append(item)

    rng = random.Random(seed)
    train_entries = []
    val_entries = []

    for label in sorted(by_label):
        items = by_label[label]
        rng.shuffle(items)
        val_count = val_counts.get(label)
        if val_count is None:
            val_count = max(1, round(len(items) * 0.2))

        classname = year_first(class_names[label])
        for item in items[:val_count]:
            val_entries.append([item["relative_path"], label, classname])
        for item in items[val_count:]:
            train_entries.append([item["relative_path"], label, classname])

    test_entries = [
        [item["relative_path"], item["label"], year_first(class_names[item["label"]])]
        for item in test_items
    ]

    return {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }


def save_annotations(path: Path, items: list[dict]) -> None:
    dtype = [
        ("bbox_x1", "O"),
        ("bbox_y1", "O"),
        ("bbox_x2", "O"),
        ("bbox_y2", "O"),
        ("class", "O"),
        ("fname", "O"),
    ]
    annotations = np.empty((1, len(items)), dtype=dtype)

    for index, item in enumerate(items):
        annotations[0, index]["bbox_x1"] = np.array([[1]], dtype=np.uint16)
        annotations[0, index]["bbox_y1"] = np.array([[1]], dtype=np.uint16)
        annotations[0, index]["bbox_x2"] = np.array([[1]], dtype=np.uint16)
        annotations[0, index]["bbox_y2"] = np.array([[1]], dtype=np.uint16)
        annotations[0, index]["class"] = np.array([[item["label"] + 1]], dtype=np.uint16)
        annotations[0, index]["fname"] = np.array([item["filename"]], dtype=object)

    savemat(path, {"annotations": annotations})


def save_class_meta(path: Path, class_names: list[str]) -> None:
    class_array = np.empty((1, len(class_names)), dtype=object)
    for index, class_name in enumerate(class_names):
        class_array[0, index] = class_name
    savemat(path, {"class_names": class_array})


def main() -> None:
    args = parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.clip_prompts, args.class_json)

    train_paths = [args.parquet_dir / name for name in TRAIN_FILES]
    test_paths = [args.parquet_dir / name for name in TEST_FILES]
    missing = [str(path) for path in train_paths + test_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet shards: {missing}")

    train_items = write_image_split(train_paths, root / "cars_train", "hf_train", args.batch_size)
    test_items = write_image_split(test_paths, root / "cars_test", "hf_test", args.batch_size)

    split = build_split(
        train_items=train_items,
        test_items=test_items,
        class_names=class_names,
        val_counts=official_val_counts(args.official_split),
        seed=args.seed,
    )

    (root / "split_zhou_StanfordCars.json").write_text(
        json.dumps(split, indent=4),
        encoding="utf-8",
    )

    if args.official_split.exists():
        shutil.copyfile(args.official_split, root / "split_zhou_StanfordCars.official_paths.json")

    devkit = root / "devkit"
    devkit.mkdir(exist_ok=True)
    save_class_meta(devkit / "cars_meta.mat", class_names)
    save_annotations(devkit / "cars_train_annos.mat", train_items)
    save_annotations(root / "cars_test_annos_withlabels.mat", test_items)

    manifest = {
        "source": "https://huggingface.co/datasets/tanganke/stanford_cars",
        "note": (
            "Images were converted from the Hugging Face parquet mirror because "
            "the Stanford official download server returned HTTP 500 during setup. "
            "split_zhou_StanfordCars.json is compatible with the local image paths "
            "and uses Zhou official val class counts when available."
        ),
        "num_train_images": len(train_items),
        "num_test_images": len(test_items),
        "split_counts": {key: len(value) for key, value in split.items()},
        "class_count": len(class_names),
        "seed": args.seed,
        "parquet_shards": [str(path) for path in train_paths + test_paths],
    }
    (root / "SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
