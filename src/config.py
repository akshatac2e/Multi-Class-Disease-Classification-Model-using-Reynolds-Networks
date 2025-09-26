from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TrainConfig:
    seed: int
    device: str
    data: Dict[str, Any]
    model: Dict[str, Any]
    optim: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    save: Dict[str, Any]

def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg)
