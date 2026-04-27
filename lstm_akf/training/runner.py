from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from lstm_akf.datasets.json_dataset import load_dataset
from lstm_akf.models import ArmorXYModelConfig, ArmorXYResidualPredictor, MultiStepSmoothL1Loss
from lstm_akf.training.checkpoint import save_checkpoint, write_json
from lstm_akf.training.plotting import plot_training_curves
from lstm_akf.training.trainer import train_one_epoch
from lstm_akf.training.validator import validate_one_epoch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.json"))


def _resolve_output_dir(output_dir: Path | None) -> Path:
    root = _project_root() / "outputs"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    for index in range(1, 1000):
        candidate = root / f"exp{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError("Unable to allocate a new experiment directory.")


def _prepare_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "checkpoints": output_dir / "checkpoints",
        "train": output_dir / "train",
        "val": output_dir / "val",
        "plots": output_dir / "val" / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _build_model_config(args: argparse.Namespace) -> ArmorXYModelConfig:
    return ArmorXYModelConfig(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_history=args.max_history,
        min_history=args.min_history,
        future_steps=args.future_steps,
        use_baseline=args.model_type == "lstm_akf",
    )


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
    defaults = _dataset_defaults()
    parser.add_argument("--dt", type=float, default=defaults["dt"])
    parser.add_argument("--process-noise", type=float, default=defaults["process_noise"])
    parser.add_argument("--measurement-noise", type=float, default=defaults["measurement_noise"])
    parser.add_argument("--initial-covariance", type=float, default=defaults["initial_covariance"])
    parser.add_argument(
        "--adaptive-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_true",
        default=defaults["adaptive_measurement_noise"],
    )
    parser.add_argument(
        "--fixed-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_false",
        help="Disable weak AKF adaptive measurement noise updates.",
    )
    parser.add_argument("--measurement-noise-min-scale", type=float, default=defaults["measurement_noise_min_scale"])
    parser.add_argument("--measurement-noise-max-scale", type=float, default=defaults["measurement_noise_max_scale"])
    parser.add_argument("--innovation-gain", type=float, default=defaults["innovation_gain"])
    parser.add_argument("--innovation-smoothing", type=float, default=defaults["innovation_smoothing"])
    parser.add_argument("--confidence-gain", type=float, default=defaults["confidence_gain"])
    parser.add_argument("--missing-process-noise-scale", type=float, default=defaults["missing_process_noise_scale"])
    parser.add_argument("--max-process-noise-scale", type=float, default=defaults["max_process_noise_scale"])


def _build_dataset_kwargs(args: argparse.Namespace) -> dict[str, float | int | bool]:
    return {
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


def _resolve_aux_weights(args: argparse.Namespace) -> tuple[float, float]:
    if args.model_type != "lstm_akf":
        return 0.0, 0.0
    return float(args.residual_aux_weight), float(args.direct_aux_weight)


def _is_improved(score: float, best_score: float, min_delta: float) -> bool:
    return score < (best_score - max(0.0, float(min_delta)))


def _build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    scheduler_name = args.scheduler.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=max(1, args.scheduler_step_size),
            gamma=args.scheduler_gamma,
        )
    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.scheduler_t_max or args.epochs),
            eta_min=args.min_lr,
        )
    if scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_gamma,
            patience=max(0, args.scheduler_patience),
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def _append_log(log_path: Path, row: dict[str, float | int | str]) -> None:
    exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _print_startup_summary(
    args: argparse.Namespace,
    output_dir: Path,
    train_size: int,
    val_size: int,
) -> None:
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Model type: {args.model_type}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Train samples: {train_size}", flush=True)
    print(f"Val samples: {val_size}", flush=True)
    print(f"Scheduler: {args.scheduler}", flush=True)
    if args.early_stopping_patience > 0:
        print(
            f"Early stopping: enabled | patience={args.early_stopping_patience} | min_delta={args.early_stopping_min_delta}",
            flush=True,
        )
    else:
        print("Early stopping: disabled", flush=True)


def _format_metric_block(prefix: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return f"{prefix}: N/A"
    return (
        f"{prefix}: loss={metrics['loss']:.6f}, "
        f"ade={metrics['ade']:.6f}, "
        f"fde={metrics['fde']:.6f}"
    )


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train the LSTM-AKF model.")
    parser.add_argument("--train-data", type=Path, default=root / "data" / "dataset" / "train")
    parser.add_argument("--val-data", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-type", choices=["lstm_akf", "lstm_only"], default="lstm_akf")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine", "plateau"], default="step")
    parser.add_argument("--scheduler-step-size", type=int, default=30)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=15)
    parser.add_argument("--future-steps", type=int, default=15)
    parser.add_argument("--residual-aux-weight", type=float, default=0.2)
    parser.add_argument("--direct-aux-weight", type=float, default=0.2)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop training after this many epochs without monitored loss improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum loss decrease required to reset early stopping patience.",
    )
    _add_akf_args(parser)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm batch progress bars.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate training curve images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    layout = _prepare_layout(output_dir)
    model_config = _build_model_config(args)
    dataset_kwargs = _build_dataset_kwargs(args)
    residual_aux_weight, direct_aux_weight = _resolve_aux_weights(args)
    device = torch.device(args.device)

    train_dataset = load_dataset(
        args.train_data,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    has_val = _count_json_files(args.val_data) > 0
    val_loader = None
    if has_val:
        val_dataset = load_dataset(
            args.val_data,
            **dataset_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
                )

    model = ArmorXYResidualPredictor(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(args, optimizer)
    loss_fn = MultiStepSmoothL1Loss(beta=args.beta)

    config_payload = {
        "model_type": args.model_type,
        "model": asdict(model_config),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_t_max": args.scheduler_t_max,
        "min_lr": args.min_lr,
        "beta": args.beta,
        "residual_aux_weight": residual_aux_weight,
        "direct_aux_weight": direct_aux_weight,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "device": args.device,
        "dataset": dataset_kwargs,
    }
    write_json(layout["train"] / "config.json", config_payload)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    _print_startup_summary(args, output_dir, len(train_dataset), val_size)

    best_score = float("inf")
    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    history_rows: list[dict[str, float | int | str]] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            residual_aux_weight=residual_aux_weight,
            direct_aux_weight=direct_aux_weight,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=not args.no_progress,
        )
        val_metrics = (
            validate_one_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                residual_aux_weight=residual_aux_weight,
                direct_aux_weight=direct_aux_weight,
                epoch=epoch,
                total_epochs=args.epochs,
                show_progress=not args.no_progress,
            )
            if val_loader is not None
            else None
        )

        monitored = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        is_best = False
        if _is_improved(monitored, best_score, args.early_stopping_min_delta):
            best_score = monitored
            best_metrics = val_metrics or train_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ade": train_metrics["ade"],
            "train_fde": train_metrics["fde"],
            "val_loss": val_metrics["loss"] if val_metrics is not None else "",
            "val_ade": val_metrics["ade"] if val_metrics is not None else "",
            "val_fde": val_metrics["fde"] if val_metrics is not None else "",
            "monitored_loss": monitored,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "lr": current_lr,
        }
        history_rows.append(log_row)
        _append_log(layout["train"] / "log.csv", log_row)
        save_checkpoint(
            layout["checkpoints"] / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=val_metrics or train_metrics,
            config=config_payload,
        )
        if is_best:
            save_checkpoint(
                layout["checkpoints"] / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=best_metrics,
                config=config_payload,
            )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(monitored)
            else:
                scheduler.step()

        if val_metrics is not None:
            write_json(layout["val"] / "metrics.json", {"epoch": epoch, **val_metrics})
        if not args.no_plot:
            plot_path = plot_training_curves(history_rows, layout["train"] / "curves.png")
            if plot_path is not None:
                print(f"Curves: {plot_path}", flush=True)

        best_mark = " [best]" if is_best else ""
        print(_format_metric_block("Train", train_metrics), flush=True)
        print(_format_metric_block("Val", val_metrics), flush=True)
        print(
            f"Checkpoint: {layout['checkpoints'] / 'last.pt'}{best_mark} | "
            f"lr={current_lr:.6g} -> next={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"Early stopping triggered at epoch {epoch} | best_epoch={best_epoch} | best_score={best_score:.6f}",
                flush=True,
            )
            break

    if best_metrics is not None:
        write_json(
            layout["train"] / "summary.json",
            {
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
                "stop_epoch": stop_epoch,
                **best_metrics,
            },
        )
        print(
            f"\nBest score: {best_score:.6f} | Summary: {layout['train'] / 'summary.json'}",
            flush=True,
        )
