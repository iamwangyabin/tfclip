from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    path: Path
    label: int
    classname: str


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    classnames: list[str]
    templates: list[str]
    train: list[Example]
    val: list[Example]
    test: list[Example]
