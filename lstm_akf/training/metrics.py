from __future__ import annotations

from typing import Sequence

import torch


def average_displacement_error(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_xy - target_xy, dim=-1).mean()


def final_displacement_error(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_xy[:, -1] - target_xy[:, -1], dim=-1).mean()


def compute_metrics(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    loss: torch.Tensor | None = None,
) -> dict[str, float]:
    metrics = {
        "ade": float(average_displacement_error(pred_xy, target_xy).detach().item()),
        "fde": float(final_displacement_error(pred_xy, target_xy).detach().item()),
    }
    if loss is not None:
        metrics["loss"] = float(loss.detach().item())
    return metrics


def point_metric_arrays(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    frame_index: int,
    hit_threshold: float = 0.02,
) -> dict[str, list[float]]:
    point_pred = pred_xy[:, frame_index, :]
    point_target = target_xy[:, frame_index, :]
    point_diff = point_pred - point_target
    mse = (point_diff ** 2).mean(dim=1)
    rmse = torch.sqrt(mse)
    mae = point_diff.abs().mean(dim=1)
    de = torch.linalg.norm(point_diff, dim=1)
    hit_rate = (de <= float(hit_threshold)).to(dtype=torch.float32)
    return {
        "de": de.detach().cpu().tolist(),
        "mse": mse.detach().cpu().tolist(),
        "rmse": rmse.detach().cpu().tolist(),
        "mae": mae.detach().cpu().tolist(),
        "hit_rate": hit_rate.detach().cpu().tolist(),
    }


def sequence_metric_arrays(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    horizon: int,
    hit_threshold: float = 0.02,
) -> dict[str, list[float]]:
    sequence_pred = pred_xy[:, :horizon, :]
    sequence_target = target_xy[:, :horizon, :]
    sequence_diff = sequence_pred - sequence_target
    frame_de = torch.linalg.norm(sequence_diff, dim=2)
    de = frame_de.mean(dim=1)
    mse = (sequence_diff ** 2).mean(dim=(1, 2))
    rmse = torch.sqrt(mse)
    mae = sequence_diff.abs().mean(dim=(1, 2))
    hit_rate = (frame_de <= float(hit_threshold)).to(dtype=torch.float32).mean(dim=1)
    return {
        "de": de.detach().cpu().tolist(),
        "mse": mse.detach().cpu().tolist(),
        "rmse": rmse.detach().cpu().tolist(),
        "mae": mae.detach().cpu().tolist(),
        "hit_rate": hit_rate.detach().cpu().tolist(),
    }


def summarize_metric_values(values: Sequence[float]) -> dict[str, float | list[float]]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "values": tensor.tolist(),
    }


def summarize_metric_arrays(metric_arrays: dict[str, Sequence[float]]) -> dict[str, dict[str, float | list[float]]]:
    return {name: summarize_metric_values(values) for name, values in metric_arrays.items()}


def weighted_average(metric_sums: dict[str, float], total_weight: int) -> dict[str, float]:
    if total_weight <= 0:
        return {key: 0.0 for key in metric_sums}
    return {key: value / total_weight for key, value in metric_sums.items()}
