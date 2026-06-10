from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "methods" / "official_feature_space.yaml"


@lru_cache(maxsize=1)
def load_official_method_configs() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def official_dataset_key(dataset: str | None) -> str:
    if dataset is None:
        return "all"

    key = dataset.lower().replace("-", "_")
    aliases = load_official_method_configs().get("dataset_aliases", {})
    return aliases.get(key, aliases.get(dataset.lower(), key))


def get_method_config(method: str, dataset: str | None = None) -> dict:
    method = method.lower()
    configs = load_official_method_configs().get(method, {})
    key = official_dataset_key(dataset)
    merged = deepcopy(configs.get("all", {}))
    if key in configs:
        merged.update(deepcopy(configs[key]))
    return merged
