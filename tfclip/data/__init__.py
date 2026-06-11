from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import DatasetBundle, Example


DATASET_NAMES = (
    "dtd",
    "food101",
    "oxford_pets",
    "flowers102",
    "fgvc_aircraft",
    "eurosat",
    "stanford_cars",
)


@dataclass(frozen=True)
class ZhouSplitSpec:
    name: str
    dataset_dir: str
    image_dir: str
    split_file: str
    templates: list[str]


ZHOU_SPLITS = {
    "dtd": ZhouSplitSpec(
        name="dtd",
        dataset_dir="dtd",
        image_dir="images",
        split_file="split_zhou_DescribableTextures.json",
        templates=["{} texture."],
    ),
    "food101": ZhouSplitSpec(
        name="food101",
        dataset_dir="food-101",
        image_dir="images",
        split_file="split_zhou_Food101.json",
        templates=["a photo of {}, a type of food."],
    ),
    "oxford_pets": ZhouSplitSpec(
        name="oxford_pets",
        dataset_dir="oxford_pets",
        image_dir="images",
        split_file="split_zhou_OxfordPets.json",
        templates=["a photo of a {}, a type of pet."],
    ),
    "flowers102": ZhouSplitSpec(
        name="flowers102",
        dataset_dir="oxford_flowers",
        image_dir="jpg",
        split_file="split_zhou_OxfordFlowers.json",
        templates=["a photo of a {}, a type of flower."],
    ),
    "eurosat": ZhouSplitSpec(
        name="eurosat",
        dataset_dir="eurosat",
        image_dir="2750",
        split_file="split_zhou_EuroSAT.json",
        templates=["a centered satellite photo of {}."],
    ),
    "stanford_cars": ZhouSplitSpec(
        name="stanford_cars",
        dataset_dir="stanford_cars",
        image_dir=".",
        split_file="split_zhou_StanfordCars.json",
        templates=["a photo of a {}."],
    ),
}

ALIASES = {
    "flowers_102": "flowers102",
    "oxford_flowers": "flowers102",
    "oxfordflowers": "flowers102",
    "flower102": "flowers102",
    "flowers": "flowers102",
    "fgvc": "fgvc_aircraft",
    "fgvcaircraft": "fgvc_aircraft",
    "food_101": "food101",
    "stanfordcars": "stanford_cars",
}


def load_dataset(name: str, data_root: str | Path = "data", seed: int | None = None) -> DatasetBundle:
    del seed
    key = _canonical_name(name)
    root = Path(data_root)
    if key == "fgvc_aircraft":
        return _load_fgvc_aircraft(root)
    if key in ZHOU_SPLITS:
        return _load_zhou_split(root, ZHOU_SPLITS[key])
    valid = ", ".join(DATASET_NAMES)
    raise KeyError(f"Unknown dataset {name!r}; expected one of: {valid}")


def _canonical_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    return ALIASES.get(key, key)


def _load_zhou_split(data_root: Path, spec: ZhouSplitSpec) -> DatasetBundle:
    dataset_dir = data_root / spec.dataset_dir
    image_dir = dataset_dir / spec.image_dir
    split_path = dataset_dir / spec.split_file
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = _convert_zhou_items(split["train"], image_dir)
    val = _convert_zhou_items(split["val"], image_dir)
    test = _convert_zhou_items(split["test"], image_dir)
    classnames = _classnames_from_splits(train, val, test)
    return DatasetBundle(
        name=spec.name,
        classnames=classnames,
        templates=spec.templates,
        train=train,
        val=val,
        test=test,
    )


def _convert_zhou_items(items: list[list[object]], image_dir: Path) -> list[Example]:
    examples = []
    for relative_path, label, classname in items:
        examples.append(
            Example(
                path=image_dir / str(relative_path),
                label=int(label),
                classname=str(classname),
            )
        )
    return examples


def _load_fgvc_aircraft(data_root: Path) -> DatasetBundle:
    dataset_dir = data_root / "fgvc_aircraft"
    variants_path = dataset_dir / "variants.txt"
    if not variants_path.exists():
        raise FileNotFoundError(f"Missing FGVC variants file: {variants_path}")

    classnames = [line.strip() for line in variants_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_by_classname = {classname: label for label, classname in enumerate(classnames)}
    return DatasetBundle(
        name="fgvc_aircraft",
        classnames=classnames,
        templates=["a photo of a {}, a type of aircraft."],
        train=_read_fgvc_split(dataset_dir, "images_variant_train.txt", label_by_classname),
        val=_read_fgvc_split(dataset_dir, "images_variant_val.txt", label_by_classname),
        test=_read_fgvc_split(dataset_dir, "images_variant_test.txt", label_by_classname),
    )


def _read_fgvc_split(dataset_dir: Path, filename: str, label_by_classname: dict[str, int]) -> list[Example]:
    split_path = dataset_dir / filename
    if not split_path.exists():
        raise FileNotFoundError(f"Missing FGVC split file: {split_path}")

    examples = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        image_id, classname = line.split(" ", 1)
        examples.append(
            Example(
                path=dataset_dir / "images" / f"{image_id}.jpg",
                label=label_by_classname[classname],
                classname=classname,
            )
        )
    return examples


def _classnames_from_splits(*splits: list[Example]) -> list[str]:
    by_label: dict[int, str] = {}
    for split in splits:
        for item in split:
            existing = by_label.setdefault(item.label, item.classname)
            if existing != item.classname:
                raise ValueError(
                    f"Label {item.label} maps to both {existing!r} and {item.classname!r}"
                )

    if not by_label:
        raise ValueError("No class labels found")
    missing = [label for label in range(max(by_label) + 1) if label not in by_label]
    if missing:
        raise ValueError(f"Missing class labels: {missing[:10]}")
    return [by_label[label] for label in range(len(by_label))]


__all__ = ["DATASET_NAMES", "DatasetBundle", "Example", "load_dataset"]
