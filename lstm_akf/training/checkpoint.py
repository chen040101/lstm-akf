from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _serialize_config(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return config


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    scheduler: Any = None,
    metrics: dict[str, float] | None = None,
    config: Any = None,
) -> Path:
    path = ensure_parent(path)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics or {},
        "config": _serialize_config(config),
        "model_config": _serialize_config(config),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
