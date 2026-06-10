#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "methods" / "official_feature_space.yaml"

TIP_FIELDS = ("search_scale", "search_step", "init_beta", "init_alpha", "augment_epoch", "train_epoch")
APE_FIELDS = (
    "search_scale",
    "search_step",
    "init_beta",
    "init_alpha",
    "init_gamma",
    "eps",
    "training_free_feat_num",
    "training_feat_num",
    "w_training_free",
    "w_training",
    "augment_epoch",
    "train_epoch",
)
LPPLUSPLUS_FIELDS = ("method", "train_epoch", "batch_size", "lr", "num_step", "augment_epoch")
PROKER_FIELDS = ("beta", "lmbda")
GDA_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalized_key(name: str) -> str:
    return name.lower().replace("-", "_")


def official_key(unified: dict[str, Any], dataset: str) -> str:
    aliases = unified.get("dataset_aliases", {})
    key = normalized_key(dataset)
    return aliases.get(key, key)


def method_config(unified: dict[str, Any], method: str, dataset: str = "all") -> dict[str, Any]:
    section = unified.get(method, {})
    merged = dict(section.get("all", {}))
    key = official_key(unified, dataset)
    merged.update(section.get(key, {}))
    return merged


def compare_fields(errors: list[str], label: str, expected: dict[str, Any], actual: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if expected.get(field) != actual.get(field):
            errors.append(f"{label}.{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}")


def audit_tip_adapter(unified: dict[str, Any], errors: list[str]) -> None:
    for path in sorted((ROOT / "external" / "baselines" / "tip_adapter" / "configs").glob("*.yaml")):
        expected = load_yaml(path)
        actual = method_config(unified, "tip_adapter", path.stem)
        compare_fields(errors, f"tip_adapter.{path.stem}", expected, actual, TIP_FIELDS)


def audit_ape(unified: dict[str, Any], errors: list[str]) -> None:
    for path in sorted((ROOT / "external" / "baselines" / "ape" / "configs").glob("*.yaml")):
        expected = load_yaml(path)
        actual = method_config(unified, "ape", path.stem)
        compare_fields(errors, f"ape.{path.stem}", expected, actual, APE_FIELDS)
        if actual.get("text_prompts") != "template_plus_cupl":
            errors.append(f"ape.{path.stem}.text_prompts: expected 'template_plus_cupl', got {actual.get('text_prompts')!r}")


def audit_lpplusplus(unified: dict[str, Any], errors: list[str]) -> None:
    expected = load_yaml(ROOT / "external" / "baselines" / "lpplusplus" / "configs" / "base.yaml")
    actual = method_config(unified, "lpplusplus")
    compare_fields(errors, "lpplusplus.all", expected, actual, LPPLUSPLUS_FIELDS)
    if actual.get("classifier_centroid") != "class_sum":
        errors.append(f"lpplusplus.all.classifier_centroid: expected 'class_sum', got {actual.get('classifier_centroid')!r}")
    if actual.get("alpha_update_interval") != 10:
        errors.append(f"lpplusplus.all.alpha_update_interval: expected 10, got {actual.get('alpha_update_interval')!r}")


def audit_gda_clip(unified: dict[str, Any], errors: list[str]) -> None:
    for path in sorted((ROOT / "external" / "baselines" / "gda_clip" / "configs" / "few_shots").glob("*.yaml")):
        expected = load_yaml(path)
        actual = method_config(unified, "gda_clip", path.stem)
        compare_fields(errors, f"gda_clip.{path.stem}", expected, actual, ("augment_epoch",))
        if actual.get("alpha_grid") != GDA_ALPHA_GRID:
            errors.append(f"gda_clip.{path.stem}.alpha_grid: expected {GDA_ALPHA_GRID!r}, got {actual.get('alpha_grid')!r}")


def audit_proker(unified: dict[str, Any], errors: list[str]) -> None:
    config_root = ROOT / "external" / "baselines" / "proker" / "configs" / "RN50" / "configs_proker"
    for path in sorted(config_root.glob("*.yaml")):
        expected = load_yaml(path)
        actual = method_config(unified, "proker", path.stem)
        compare_fields(errors, f"proker.{path.stem}", expected, actual, PROKER_FIELDS)
        if actual.get("augment_epoch") != 10:
            errors.append(f"proker.{path.stem}.augment_epoch: expected 10, got {actual.get('augment_epoch')!r}")


def main() -> int:
    unified = load_yaml(CONFIG)
    errors: list[str] = []

    for label, path in unified.get("source_snapshots", {}).items():
        if not (ROOT / path).exists():
            errors.append(f"source_snapshots.{label}: missing path {path}")

    audit_tip_adapter(unified, errors)
    audit_ape(unified, errors)
    audit_lpplusplus(unified, errors)
    audit_gda_clip(unified, errors)
    audit_proker(unified, errors)

    if errors:
        print("Calibration audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Calibration audit passed: unified method defaults match vendored official configs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
