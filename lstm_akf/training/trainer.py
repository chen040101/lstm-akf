from __future__ import annotations

import sys

import torch

from lstm_akf.models import compute_model_loss
from lstm_akf.training.metrics import compute_metrics, weighted_average

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    residual_aux_weight: float = 0.0,
    direct_aux_weight: float = 0.0,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    metric_sums = {"loss": 0.0, "ade": 0.0, "fde": 0.0}
    total_items = 0

    iterator = dataloader
    progress = None
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if show_progress and tqdm is not None:
        if epoch is not None and total_epochs is not None:
            description = f"Train {epoch}/{total_epochs}"
        else:
            description = "Train"
        progress = tqdm(
            dataloader,
            desc=description,
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )
        iterator = progress

    for batch_index, batch in enumerate(iterator, start=1):
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            history_xy=batch["history_xy"],
            baseline_xy=batch.get("baseline_xy"),
            history_len=batch.get("history_len"),
        )
        loss = compute_model_loss(
            outputs,
            batch["target_xy"],
            loss_fn,
            residual_aux_weight=residual_aux_weight,
            direct_aux_weight=direct_aux_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(batch["history_xy"].shape[0])
        metrics = compute_metrics(outputs["pred_xy"], batch["target_xy"], loss)
        for key, value in metrics.items():
            metric_sums[key] += value * batch_size
        total_items += batch_size
        average_metrics = weighted_average(metric_sums, total_items)
        if progress is not None:
            progress.set_postfix(
                loss=f"{average_metrics['loss']:.6f}",
                ade=f"{average_metrics['ade']:.6f}",
                fde=f"{average_metrics['fde']:.6f}",
            )
        elif show_progress and total_batches is not None:
            print(
                f"  [train {batch_index}/{total_batches}] "
                f"loss={average_metrics['loss']:.6f} "
                f"ade={average_metrics['ade']:.6f} "
                f"fde={average_metrics['fde']:.6f}",
                flush=True,
            )

    if progress is not None:
        progress.close()

    return weighted_average(metric_sums, total_items)
