from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data.types import DatasetBundle, Example
from .models.openclip import OpenCLIPBackbone


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

APE_CUPL_PROMPTS = {
    "caltech101": "CuPL_prompts_caltech101.json",
    "dtd": "CuPL_prompts_dtd.json",
    "eurosat": "CuPL_prompts_eurosat.json",
    "fgvc": "CuPL_prompts_fgvcaircraft.json",
    "fgvc_aircraft": "CuPL_prompts_fgvcaircraft.json",
    "flowers102": "CuPL_prompts_flowers102.json",
    "food101": "CuPL_prompts_food101.json",
    "imagenet": "CuPL_prompts_imagenet.json",
    "oxford_flowers": "CuPL_prompts_flowers102.json",
    "oxford_pets": "CuPL_prompts_oxfordpets.json",
    "stanford_cars": "CuPL_prompts_stanfordcars.json",
    "sun397": "CuPL_prompts_sun397.json",
    "ucf101": "CuPL_prompts_ucf101.json",
}

CUPL_CLASS_ALIASES = {
    "air plant": "bromelia",
    "globe flower": "globe-flower",
    "pink and yellow dahlia": "pink-yellow dahlia",
    "annual crop land": "Annual Crop Land",
    "forest": "Forest",
    "brushland or shrubland": "Herbaceous Vegetation Land",
    "herbaceous vegetation land": "Herbaceous Vegetation Land",
    "highway or road": "Highway or Road",
    "industrial buildings": "Industrial Buildings",
    "industrial buildings or commercial buildings": "Industrial Buildings",
    "lake or sea": "Sea or Lake",
    "pasture land": "Pasture Land",
    "permanent crop land": "Permanent Crop Land",
    "residential buildings": "Residential Buildings",
    "residential buildings or homes or apartments": "Residential Buildings",
    "river": "River",
    "sea or lake": "Sea or Lake",
}


@dataclass(frozen=True)
class FeatureSplit:
    features: np.ndarray
    labels: np.ndarray
    paths: list[str]


@dataclass(frozen=True)
class FeatureBundle:
    dataset: str
    model_id: str
    classnames: list[str]
    templates: list[str]
    train: FeatureSplit
    val: FeatureSplit
    test: FeatureSplit


class ImageExampleDataset(Dataset):
    def __init__(self, examples: list[Example], preprocess):
        self.examples = examples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        item = self.examples[index]
        with Image.open(item.path) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, item.label, str(item.path)


def _extract_split(
    backbone: OpenCLIPBackbone,
    examples: list[Example],
    batch_size: int,
    num_workers: int,
) -> FeatureSplit:
    loader = DataLoader(
        ImageExampleDataset(examples, backbone.preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features, labels, paths = [], [], []
    for images, target, path in tqdm(loader, desc="features", leave=False):
        features.append(backbone.encode_images(images).cpu().numpy())
        labels.append(target.numpy())
        paths.extend(path)

    return FeatureSplit(
        features=np.concatenate(features, axis=0).astype("float32"),
        labels=np.concatenate(labels, axis=0).astype("int64"),
        paths=paths,
    )


def extract_examples(
    backbone: OpenCLIPBackbone,
    examples: list[Example],
    batch_size: int,
    num_workers: int,
    train_preprocess: bool = False,
    preprocess=None,
    normalize_features: bool = True,
) -> FeatureSplit:
    preprocess = preprocess or (backbone.train_preprocess if train_preprocess else backbone.preprocess)
    loader = DataLoader(
        ImageExampleDataset(examples, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    features, labels, paths = [], [], []
    for images, target, path in tqdm(loader, desc="support", leave=False):
        features.append(backbone.encode_images(images, normalize=normalize_features).cpu().numpy())
        labels.append(target.numpy())
        paths.extend(path)
    return FeatureSplit(
        features=np.concatenate(features, axis=0).astype("float32"),
        labels=np.concatenate(labels, axis=0).astype("int64"),
        paths=paths,
    )


def augmented_support_cache_path(
    cache_root: str | Path,
    dataset: str,
    model_id: str,
    seed: int,
    shots: int,
    augment_epoch: int,
    transform_id: str = "official_openai_clip_train",
) -> Path:
    safe_transform = re.sub(r"[^A-Za-z0-9_.-]+", "-", transform_id).strip("-")
    return Path(cache_root) / dataset / f"{model_id}__support_s{seed}_k{shots}_aug{augment_epoch}_{safe_transform}_raw.npz"


def official_openai_clip_train_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ]
    )


def support_preprocess(backbone: OpenCLIPBackbone, policy: str):
    if policy == "official_openai_clip_train":
        return official_openai_clip_train_preprocess()
    if policy == "open_clip_train":
        return backbone.train_preprocess
    if policy == "open_clip_eval":
        return backbone.preprocess
    raise ValueError(f"Unknown support preprocess policy {policy!r}")


def extract_augmented_support(
    backbone: OpenCLIPBackbone,
    examples: list[Example],
    cache_file: str | Path,
    augment_epoch: int,
    batch_size: int,
    num_workers: int,
    preprocess_policy: str = "official_openai_clip_train",
    seed: int | None = None,
) -> FeatureSplit:
    cache_file = Path(cache_file)
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return FeatureSplit(
            data["features"],
            data["labels"],
            [str(x) for x in data["paths"].tolist()],
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    preprocess = support_preprocess(backbone, preprocess_policy)
    all_features, all_labels, all_paths = [], [], []
    for _ in range(augment_epoch):
        split = extract_examples(
            backbone,
            examples,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            normalize_features=False,
        )
        all_features.append(split.features)
        all_labels.append(split.labels)
        all_paths.extend(split.paths)

    features = np.concatenate(all_features, axis=0).astype("float32")
    labels = np.concatenate(all_labels, axis=0).astype("int64")
    np.savez_compressed(cache_file, features=features, labels=labels, paths=np.array(all_paths, dtype=object))
    return FeatureSplit(features, labels, all_paths)


def cache_path(cache_root: str | Path, dataset: str, model_id: str, seed: int | None = None) -> Path:
    suffix = f"__split_s{seed}" if seed is not None else ""
    return Path(cache_root) / dataset / f"{model_id}{suffix}.npz"


def save_feature_bundle(bundle: FeatureBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset=bundle.dataset,
        model_id=bundle.model_id,
        classnames=np.array(bundle.classnames, dtype=object),
        templates=np.array(bundle.templates, dtype=object),
        train_features=bundle.train.features,
        train_labels=bundle.train.labels,
        train_paths=np.array(bundle.train.paths, dtype=object),
        val_features=bundle.val.features,
        val_labels=bundle.val.labels,
        val_paths=np.array(bundle.val.paths, dtype=object),
        test_features=bundle.test.features,
        test_labels=bundle.test.labels,
        test_paths=np.array(bundle.test.paths, dtype=object),
    )
    metadata = {
        "dataset": bundle.dataset,
        "model_id": bundle.model_id,
        "classes": len(bundle.classnames),
        "train": len(bundle.train.labels),
        "val": len(bundle.val.labels),
        "test": len(bundle.test.labels),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_feature_bundle(path: str | Path) -> FeatureBundle:
    data = np.load(path, allow_pickle=True)
    return FeatureBundle(
        dataset=str(data["dataset"]),
        model_id=str(data["model_id"]),
        classnames=[str(x) for x in data["classnames"].tolist()],
        templates=[str(x) for x in data["templates"].tolist()],
        train=FeatureSplit(data["train_features"], data["train_labels"], [str(x) for x in data["train_paths"].tolist()]),
        val=FeatureSplit(data["val_features"], data["val_labels"], [str(x) for x in data["val_paths"].tolist()]),
        test=FeatureSplit(data["test_features"], data["test_labels"], [str(x) for x in data["test_paths"].tolist()]),
    )


def extract_feature_bundle(
    dataset: DatasetBundle,
    backbone: OpenCLIPBackbone,
    batch_size: int = 128,
    num_workers: int = 4,
) -> FeatureBundle:
    return FeatureBundle(
        dataset=dataset.name,
        model_id=backbone.id,
        classnames=dataset.classnames,
        templates=dataset.templates,
        train=_extract_split(backbone, dataset.train, batch_size, num_workers),
        val=_extract_split(backbone, dataset.val, batch_size, num_workers),
        test=_extract_split(backbone, dataset.test, batch_size, num_workers),
    )


@torch.no_grad()
def text_classifier(
    backbone: OpenCLIPBackbone,
    classnames: list[str],
    templates: list[str],
    batch_size: int = 256,
    extra_prompts_by_class: dict[str, list[str]] | None = None,
    require_extra_prompts: bool = False,
) -> np.ndarray:
    if require_extra_prompts and not extra_prompts_by_class:
        raise KeyError("Extra prompts are required but no prompt table was provided")

    weights = []
    for classname in classnames:
        normalized_classname = classname.replace("_", " ")
        prompts = [template.format(normalized_classname) for template in templates]
        if extra_prompts_by_class is not None:
            extra_prompts = extra_prompts_for_class(extra_prompts_by_class, normalized_classname)
            if require_extra_prompts and not extra_prompts:
                raise KeyError(f"No extra prompts found for class {normalized_classname!r}")
            prompts.extend(extra_prompts)
        text_features = backbone.encode_texts(prompts, batch_size=batch_size)
        text_features = torch.nn.functional.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
        weights.append(text_features.squeeze(0).numpy())
    return np.stack(weights).astype("float32")


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, eps)


def load_ape_cupl_prompts(dataset: str) -> dict[str, list[str]]:
    key = dataset.lower().replace("-", "_")
    filename = APE_CUPL_PROMPTS.get(key)
    if filename is None:
        return {}

    path = Path(__file__).resolve().parents[1] / "external" / "baselines" / "ape" / "gpt3_prompts" / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extra_prompts_for_class(extra_prompts_by_class: dict[str, list[str]], classname: str) -> list[str]:
    lower_classname = classname.lower()
    candidates = [
        classname,
        lower_classname,
        classname.replace("_", " "),
        classname.replace("_", " ").lower(),
        CUPL_CLASS_ALIASES.get(lower_classname, lower_classname),
        _stanford_cars_year_first(classname),
    ]
    for candidate in candidates:
        if candidate in extra_prompts_by_class:
            return list(extra_prompts_by_class[candidate])
    return []


def _stanford_cars_year_first(classname: str) -> str:
    match = re.match(r"^(.*)\s+(\d{4})$", classname)
    if not match:
        return classname
    return f"{match.group(2)} {match.group(1)}"
