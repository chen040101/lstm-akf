from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lstm_akf.datasets.json_dataset import load_dataset
from lstm_akf.models import ArmorXYModelConfig, ArmorXYResidualPredictor, MultiStepSmoothL1Loss, compute_model_loss
from lstm_akf.training.checkpoint import load_checkpoint, write_json
from lstm_akf.training.metrics import compute_metrics, weighted_average

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest_best_checkpoint(outputs_root: Path) -> Path | None:
    candidates = sorted(outputs_root.glob("exp*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _dataset_defaults() -> dict[str, float | int | bool]:
    return {
        "min_history": 5,
        "max_history": 15,
        "future_steps": 15,
        "dt": 1.0,
        "process_noise": 1e-4,
        "measurement_noise": 1e-3,
        "initial_covariance": 1.0,
        "adaptive_measurement_noise": True,
        "measurement_noise_min_scale": 0.25,
        "measurement_noise_max_scale": 16.0,
        "innovation_gain": 1.0,
        "innovation_smoothing": 0.2,
        "confidence_gain": 1.0,
        "missing_process_noise_scale": 1.35,
        "max_process_noise_scale": 6.0,
    }


def _add_akf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-history", type=int, default=None)
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument("--future-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--process-noise", type=float, default=None)
    parser.add_argument("--measurement-noise", type=float, default=None)
    parser.add_argument("--initial-covariance", type=float, default=None)
    parser.add_argument(
        "--adaptive-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--fixed-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_false",
        help="Disable weak AKF adaptive measurement noise updates.",
    )
    parser.add_argument("--measurement-noise-min-scale", type=float, default=None)
    parser.add_argument("--measurement-noise-max-scale", type=float, default=None)
    parser.add_argument("--innovation-gain", type=float, default=None)
    parser.add_argument("--innovation-smoothing", type=float, default=None)
    parser.add_argument("--confidence-gain", type=float, default=None)
    parser.add_argument("--missing-process-noise-scale", type=float, default=None)
    parser.add_argument("--max-process-noise-scale", type=float, default=None)


def _extract_model_config(checkpoint: dict) -> ArmorXYModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(raw_config, dict) and isinstance(raw_config.get("model"), dict):
        raw_config = raw_config["model"]
    return ArmorXYModelConfig(
        **{
            key: raw_config[key]
            for key in ArmorXYModelConfig.__dataclass_fields__
            if key in raw_config
        }
    )


def _resolve_dataset_kwargs(
    args: argparse.Namespace,
    checkpoint: dict,
    model_config: ArmorXYModelConfig,
) -> dict[str, float | int | bool]:
    defaults = _dataset_defaults()
    resolved: dict[str, float | int | bool] = {
        "min_history": model_config.min_history,
        "max_history": model_config.max_history,
        "future_steps": model_config.future_steps,
        "dt": defaults["dt"],
        "process_noise": defaults["process_noise"],
        "measurement_noise": defaults["measurement_noise"],
        "initial_covariance": defaults["initial_covariance"],
        "adaptive_measurement_noise": defaults["adaptive_measurement_noise"],
        "measurement_noise_min_scale": defaults["measurement_noise_min_scale"],
        "measurement_noise_max_scale": defaults["measurement_noise_max_scale"],
        "innovation_gain": defaults["innovation_gain"],
        "innovation_smoothing": defaults["innovation_smoothing"],
        "confidence_gain": defaults["confidence_gain"],
        "missing_process_noise_scale": defaults["missing_process_noise_scale"],
        "max_process_noise_scale": defaults["max_process_noise_scale"],
    }
    raw_config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(raw_config, dict) and isinstance(raw_config.get("dataset"), dict):
        resolved.update(raw_config["dataset"])

    cli_overrides = {
        "min_history": args.min_history,
        "max_history": args.max_history,
        "future_steps": args.future_steps,
        "dt": args.dt,
        "process_noise": args.process_noise,
        "measurement_noise": args.measurement_noise,
        "initial_covariance": args.initial_covariance,
        "adaptive_measurement_noise": args.adaptive_measurement_noise,
        "measurement_noise_min_scale": args.measurement_noise_min_scale,
        "measurement_noise_max_scale": args.measurement_noise_max_scale,
        "innovation_gain": args.innovation_gain,
        "innovation_smoothing": args.innovation_smoothing,
        "confidence_gain": args.confidence_gain,
        "missing_process_noise_scale": args.missing_process_noise_scale,
        "max_process_noise_scale": args.max_process_noise_scale,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def _resolve_aux_weights(
    args: argparse.Namespace,
    checkpoint: dict,
    model_config: ArmorXYModelConfig,
) -> tuple[float, float]:
    resolved = {"residual_aux_weight": 0.0, "direct_aux_weight": 0.0}
    raw_config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(raw_config, dict):
        resolved["residual_aux_weight"] = float(raw_config.get("residual_aux_weight", 0.0))
        resolved["direct_aux_weight"] = float(raw_config.get("direct_aux_weight", 0.0))

    if not model_config.use_baseline:
        return 0.0, 0.0
    return resolved["residual_aux_weight"], resolved["direct_aux_weight"]


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    loss_fn,
    device: torch.device,
    residual_aux_weight: float = 0.0,
    direct_aux_weight: float = 0.0,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    metric_sums = {"loss": 0.0, "ade": 0.0, "fde": 0.0}
    total_items = 0

    iterator = dataloader
    progress = None
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if show_progress and tqdm is not None:
        if epoch is not None and total_epochs is not None:
            description = f"Val   {epoch}/{total_epochs}"
        else:
            description = "Validate"
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
                f"  [val {batch_index}/{total_batches}] "
                f"loss={average_metrics['loss']:.6f} "
                f"ade={average_metrics['ade']:.6f} "
                f"fde={average_metrics['fde']:.6f}",
                flush=True,
            )

    if progress is not None:
        progress.close()

    return weighted_average(metric_sums, total_items)


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device: torch.device,
    residual_aux_weight: float = 0.0,
    direct_aux_weight: float = 0.0,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    return evaluate(
        model,
        dataloader,
        loss_fn,
        device,
        residual_aux_weight=residual_aux_weight,
        direct_aux_weight=direct_aux_weight,
        epoch=epoch,
        total_epochs=total_epochs,
        show_progress=show_progress,
    )


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Validate a trained LSTM-AKF model.")
    parser.add_argument("--dataset", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    _add_akf_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or _latest_best_checkpoint(_project_root() / "outputs")
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint was provided and no best.pt was found under outputs/.")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = _extract_model_config(checkpoint)
    model = ArmorXYResidualPredictor(model_config)
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device(args.device)
    model.to(device)
    dataset_kwargs = _resolve_dataset_kwargs(args, checkpoint, model_config)
    residual_aux_weight, direct_aux_weight = _resolve_aux_weights(args, checkpoint, model_config)
    dataset = load_dataset(args.dataset, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    loss_fn = MultiStepSmoothL1Loss(beta=args.beta)
    metrics = evaluate(
        model,
        dataloader,
        loss_fn,
        device,
        residual_aux_weight=residual_aux_weight,
        direct_aux_weight=direct_aux_weight,
    )

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.resolve().parents[1] / "val" / "metrics.json"
    write_json(output_path, metrics)


__all__ = ["evaluate", "main", "parse_args", "validate_one_epoch"]
