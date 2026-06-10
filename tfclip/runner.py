from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_dataset
from .features import (
    augmented_support_cache_path,
    cache_path,
    extract_augmented_support,
    extract_feature_bundle,
    load_ape_cupl_prompts,
    load_feature_bundle,
    save_feature_bundle,
    text_classifier,
)
from .methods.config import get_method_config
from .methods import run_feature_method
from .models import OpenCLIPBackbone


SUPPORT_CACHE_METHODS = {"tip_adapter", "ape", "lpplusplus", "gda_clip", "proker"}


def sample_indices_per_class(labels: np.ndarray, shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        if len(idx) < shots:
            raise ValueError(f"Class {label} has {len(idx)} train samples, fewer than shots={shots}")
        selected.extend(idx[:shots].tolist())
    return np.array(selected, dtype=np.int64)


def ensure_image_features(
    dataset_name: str,
    data_root: str | Path,
    cache_root: str | Path,
    backbone: OpenCLIPBackbone,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    dataset = load_dataset(dataset_name, data_root, seed=seed)
    path = cache_path(cache_root, dataset.name, backbone.id, seed=seed)
    if path.exists():
        return dataset, load_feature_bundle(path)

    bundle = extract_feature_bundle(dataset, backbone, batch_size=batch_size, num_workers=num_workers)
    save_feature_bundle(bundle, path)
    return dataset, bundle


def run_one(
    dataset_name: str,
    method: str,
    shots: int,
    seed: int,
    model_name: str,
    pretrained: str,
    data_root: str | Path = "data",
    cache_root: str | Path = "outputs/features",
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
) -> dict:
    backbone = OpenCLIPBackbone(model_name=model_name, pretrained=pretrained, device=device)
    dataset, features = ensure_image_features(
        dataset_name=dataset_name,
        data_root=data_root,
        cache_root=cache_root,
        backbone=backbone,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    support_idx = sample_indices_per_class(features.train.labels, shots=shots, seed=seed)
    support_examples = [dataset.train[int(index)] for index in support_idx]
    method_cfg = get_method_config(method, dataset.name)
    augmented = None
    if method in SUPPORT_CACHE_METHODS:
        augment_epoch = int(method_cfg.get("augment_epoch", 1))
        transform_id = method_cfg.get("support_transform", "official_openai_clip_train")
        augmented = extract_augmented_support(
            backbone=backbone,
            examples=support_examples,
            cache_file=augmented_support_cache_path(cache_root, dataset.name, backbone.id, seed, shots, augment_epoch, transform_id),
            augment_epoch=augment_epoch,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess_policy=transform_id,
            seed=seed,
        )
    if method == "zero_shot":
        text_features = text_classifier(backbone, dataset.classnames, ["a photo of a {}."])
    elif method == "ape" and method_cfg.get("text_prompts") == "template_plus_cupl":
        text_features = text_classifier(
            backbone,
            dataset.classnames,
            dataset.templates,
            extra_prompts_by_class=load_ape_cupl_prompts(dataset.name),
            require_extra_prompts=True,
        )
    else:
        text_features = text_classifier(backbone, dataset.classnames, dataset.templates)
    result = run_feature_method(
        method=method,
        train_x=features.train.features[support_idx],
        train_y=features.train.labels[support_idx],
        val_x=features.val.features,
        val_y=features.val.labels,
        test_x=features.test.features,
        test_y=features.test.labels,
        text_features=text_features,
        dataset=dataset.name,
        method_config={"seed": seed, "shots": shots},
        augmented_train_x=augmented.features if augmented is not None else None,
        augmented_train_y=augmented.labels if augmented is not None else None,
    )

    return {
        "dataset": dataset.name,
        "method": result.method,
        "shots": shots,
        "seed": seed,
        "model_name": model_name,
        "pretrained": pretrained,
        "model_id": backbone.id,
        "accuracy": result.accuracy,
        "best_params": result.best_params,
        "num_classes": len(dataset.classnames),
        "num_support": int(len(support_idx)),
        "num_val": int(len(features.val.labels)),
        "num_test": int(len(features.test.labels)),
    }


def run_many(
    dataset_name: str,
    methods: list[str],
    shots: int,
    seed: int,
    model_name: str,
    pretrained: str,
    data_root: str | Path = "data",
    cache_root: str | Path = "outputs/features",
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
) -> list[dict]:
    backbone = OpenCLIPBackbone(model_name=model_name, pretrained=pretrained, device=device)
    dataset, features = ensure_image_features(
        dataset_name=dataset_name,
        data_root=data_root,
        cache_root=cache_root,
        backbone=backbone,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    support_idx = sample_indices_per_class(features.train.labels, shots=shots, seed=seed)
    support_examples = [dataset.train[int(index)] for index in support_idx]
    augmented_by_key = {}
    for method in methods:
        if method not in SUPPORT_CACHE_METHODS:
            continue
        method_cfg = get_method_config(method, dataset.name)
        augment_epoch = int(method_cfg.get("augment_epoch", 1))
        transform_id = method_cfg.get("support_transform", "official_openai_clip_train")
        cache_key = (augment_epoch, transform_id)
        if cache_key in augmented_by_key:
            continue
        augmented_by_key[cache_key] = extract_augmented_support(
            backbone=backbone,
            examples=support_examples,
            cache_file=augmented_support_cache_path(cache_root, dataset.name, backbone.id, seed, shots, augment_epoch, transform_id),
            augment_epoch=augment_epoch,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess_policy=transform_id,
            seed=seed,
        )
    text_cache: dict[str, np.ndarray] = {}
    records = []

    for method in methods:
        method_cfg = get_method_config(method, dataset.name)
        if method == "zero_shot":
            template_key = "simple"
        elif method == "ape" and method_cfg.get("text_prompts") == "template_plus_cupl":
            template_key = "ape_cupl"
        else:
            template_key = "ensemble"
        if template_key not in text_cache:
            if template_key == "simple":
                text_cache[template_key] = text_classifier(backbone, dataset.classnames, ["a photo of a {}."])
            elif template_key == "ape_cupl":
                text_cache[template_key] = text_classifier(
                    backbone,
                    dataset.classnames,
                    dataset.templates,
                    extra_prompts_by_class=load_ape_cupl_prompts(dataset.name),
                    require_extra_prompts=True,
                )
            else:
                text_cache[template_key] = text_classifier(backbone, dataset.classnames, dataset.templates)
        augment_epoch = int(method_cfg.get("augment_epoch", 1))
        transform_id = method_cfg.get("support_transform", "official_openai_clip_train")
        augmented = augmented_by_key.get((augment_epoch, transform_id)) if method in SUPPORT_CACHE_METHODS else None

        result = run_feature_method(
            method=method,
            train_x=features.train.features[support_idx],
            train_y=features.train.labels[support_idx],
            val_x=features.val.features,
            val_y=features.val.labels,
            test_x=features.test.features,
            test_y=features.test.labels,
            text_features=text_cache[template_key],
            dataset=dataset.name,
            method_config={"seed": seed, "shots": shots},
            augmented_train_x=augmented.features if augmented is not None else None,
            augmented_train_y=augmented.labels if augmented is not None else None,
        )
        records.append(
            {
                "dataset": dataset.name,
                "method": result.method,
                "shots": shots,
                "seed": seed,
                "model_name": model_name,
                "pretrained": pretrained,
                "model_id": backbone.id,
                "accuracy": result.accuracy,
                "best_params": result.best_params,
                "num_classes": len(dataset.classnames),
                "num_support": int(len(support_idx)),
                "num_val": int(len(features.val.labels)),
                "num_test": int(len(features.test.labels)),
            }
        )

    return records


def append_result(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
