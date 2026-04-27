from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from lstm_akf.datasets.json_dataset import load_dataset
from lstm_akf.models import ArmorXYModelConfig, ArmorXYResidualPredictor, MultiStepSmoothL1Loss
from lstm_akf.training.checkpoint import load_checkpoint, save_checkpoint, write_json
from lstm_akf.training.metrics import point_metric_arrays, sequence_metric_arrays, summarize_metric_arrays
from lstm_akf.training.plotting import plot_metric_grid, plot_training_curves
from lstm_akf.training.trainer import train_one_epoch
from lstm_akf.training.validator import validate_one_epoch


DEFAULT_POINT_FRAMES = [8, 15]
DEFAULT_SEQUENCE_HORIZONS = [8, 15]
DEFAULT_COMPARE_EPOCHS = 50
DEFAULT_COMPARE_BATCH_SIZE = 64
DEFAULT_RESIDUAL_AUX_WEIGHT = 0.2
DEFAULT_DIRECT_AUX_WEIGHT = 0.2
DEFAULT_HIT_THRESHOLD = 0.02
DEFAULT_MODEL_ORDER = [
    "lstm_akf",
    "akf_only",
    "lstm_only",
    "ann",
    "svr",
    "ar",
    "dt",
    "knn",
]


@dataclass
class DatasetArrays:
    history_xy: np.ndarray
    baseline_xy: np.ndarray
    target_xy: np.ndarray
    history_len: np.ndarray

    @property
    def features(self) -> np.ndarray:
        return self.history_xy.reshape(self.history_xy.shape[0], -1)

    @property
    def target_flat(self) -> np.ndarray:
        return self.target_xy.reshape(self.target_xy.shape[0], -1)


class AKFOnlyPredictor(torch.nn.Module):
    def __init__(self, future_steps: int = 15) -> None:
        super().__init__()
        self.future_steps = int(future_steps)

    def forward(
        self,
        history_xy: torch.Tensor,
        baseline_xy: torch.Tensor | None = None,
        history_len: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del history_xy, history_len
        if baseline_xy is None:
            raise ValueError("AKF-only predictor expects baseline_xy in the batch.")
        return {
            "pred_xy": baseline_xy,
            "pred_delta_xy": torch.zeros_like(baseline_xy),
            "baseline_xy": baseline_xy,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_output_dir(output_dir: Path | None) -> Path:
    root = output_dir or (_project_root() / "Baseline Models")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prepare_model_layout(root_dir: Path, model_slug: str) -> dict[str, Path]:
    model_root = root_dir / model_slug
    paths = {
        "root": model_root,
        "artifacts": model_root / "artifacts",
        "checkpoints": model_root / "checkpoints",
        "train": model_root / "train",
        "val": model_root / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _prepare_comparison_layout(root_dir: Path) -> dict[str, Path]:
    comparison_root = root_dir / "comparison"
    docs_root = comparison_root / "summaries"
    comparison_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": comparison_root,
        "docs": docs_root,
        "summary": comparison_root / "summary_results.json",
        "detailed": comparison_root / "detailed_results.json",
        "csv": comparison_root / "comparison.csv",
    }


def _normalize_indices(values: list[int], upper_bound: int, kind: str) -> list[int]:
    normalized = sorted(set(int(value) for value in values))
    for value in normalized:
        if value <= 0 or value > upper_bound:
            raise ValueError(f"{kind} value must be in [1, {upper_bound}], got {value}.")
    return normalized


def _count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.json"))


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


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


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _serialize_estimator_params(estimator: Any) -> dict[str, Any]:
    if hasattr(estimator, "get_params"):
        return {key: repr(value) for key, value in estimator.get_params(deep=True).items()}
    return {"repr": repr(estimator)}


def _build_model_config(use_baseline: bool) -> ArmorXYModelConfig:
    return ArmorXYModelConfig(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        min_history=5,
        max_history=15,
        future_steps=15,
        use_baseline=use_baseline,
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


def _build_dataset_kwargs(
    args: argparse.Namespace,
    model_config: ArmorXYModelConfig,
) -> dict[str, float | int | bool]:
    return {
        "min_history": model_config.min_history,
        "max_history": model_config.max_history,
        "future_steps": model_config.future_steps,
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


def _load_dataset_objects(
    train_path: Path,
    val_path: Path,
    batch_size: int,
    num_workers: int,
    dataset_kwargs: dict[str, float | int | bool],
) -> tuple[Any, DataLoader, Any, DataLoader]:
    train_dataset = load_dataset(
        train_path,
        **dataset_kwargs,
    )
    val_dataset = load_dataset(
        val_path,
        **dataset_kwargs,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, train_loader, val_dataset, val_loader


def _dataset_to_arrays(dataset: Any) -> DatasetArrays:
    history_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    history_len_rows: list[np.ndarray] = []
    for index in range(len(dataset)):
        item = dataset[index]
        history_rows.append(item["history_xy"].detach().cpu().numpy())
        baseline_rows.append(item["baseline_xy"].detach().cpu().numpy())
        target_rows.append(item["target_xy"].detach().cpu().numpy())
        history_len_rows.append(item["history_len"].detach().cpu().numpy())
    return DatasetArrays(
        history_xy=np.stack(history_rows, axis=0).astype(np.float32),
        baseline_xy=np.stack(baseline_rows, axis=0).astype(np.float32),
        target_xy=np.stack(target_rows, axis=0).astype(np.float32),
        history_len=np.asarray(history_len_rows, dtype=np.int64).reshape(-1),
    )


def _dataset_fingerprint(dataset: Any) -> dict[str, Any]:
    sample_files = getattr(dataset, "sample_files", None)
    if sample_files is None:
        return {
            "sample_count": int(len(dataset)),
            "sha256": "",
        }
    digest = hashlib.sha256()
    for sample_path in sample_files:
        digest.update(str(Path(sample_path).name).encode("utf-8"))
        digest.update(b"\n")
    return {
        "sample_count": int(len(sample_files)),
        "sha256": digest.hexdigest(),
    }


def _extract_model_config(checkpoint: dict[str, Any]) -> ArmorXYModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(raw_config, dict) and isinstance(raw_config.get("model"), dict):
        raw_config = raw_config["model"]
    fields = ArmorXYModelConfig.__dataclass_fields__
    payload = {key: raw_config[key] for key in fields if key in raw_config}
    return ArmorXYModelConfig(**payload)


def _load_trained_torch_model(checkpoint_path: Path, device: torch.device) -> tuple[ArmorXYResidualPredictor, ArmorXYModelConfig]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = _extract_model_config(checkpoint)
    model = ArmorXYResidualPredictor(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, model_config


def _save_pickle(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _predict_torch_model(
    model: torch.nn.Module,
    arrays: DatasetArrays,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    predictions: list[np.ndarray] = []
    inference_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(arrays.history_xy), batch_size):
            end = min(start + batch_size, len(arrays.history_xy))
            history_xy = torch.from_numpy(arrays.history_xy[start:end]).to(device)
            baseline_xy = torch.from_numpy(arrays.baseline_xy[start:end]).to(device)
            history_len = torch.from_numpy(arrays.history_len[start:end]).to(device)
            _sync_device(device)
            start_time = time.perf_counter()
            output = model(history_xy=history_xy, baseline_xy=baseline_xy, history_len=history_len)["pred_xy"]
            _sync_device(device)
            elapsed = time.perf_counter() - start_time
            per_sample = elapsed / max(1, end - start)
            inference_values.extend([per_sample] * (end - start))
            predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32), inference_values


def _predict_classical_model(
    estimator: Any,
    features: np.ndarray,
    future_steps: int,
) -> tuple[np.ndarray, list[float]]:
    start_time = time.perf_counter()
    pred_flat = estimator.predict(features)
    elapsed = time.perf_counter() - start_time
    pred_flat = np.asarray(pred_flat, dtype=np.float32)
    predictions = pred_flat.reshape(pred_flat.shape[0], future_steps, 2)
    per_sample = elapsed / max(1, len(features))
    return predictions, [per_sample] * len(features)


def _fit_classical_estimator(model_slug: str) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    if model_slug == "ann":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        batch_size=64,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=12,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_slug == "svr":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.01))),
            ]
        )
    if model_slug == "ar":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    if model_slug == "dt":
        return DecisionTreeRegressor(max_depth=12, random_state=42)
    if model_slug == "knn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
            ]
        )
    raise ValueError(f"Unsupported classical model: {model_slug}")


def _build_training_history_from_loss_curve(loss_curve: list[float]) -> list[dict[str, float | int | str]]:
    return [
        {
            "epoch": index + 1,
            "train_loss": float(loss_value),
            "val_loss": "",
            "lr": "",
        }
        for index, loss_value in enumerate(loss_curve)
    ]


def _load_external_lstm_akf_model(
    root_dir: Path,
    device: torch.device,
    dataset_config: dict[str, float | int | bool],
    dataset_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    model_name = "LSTM-AKF"
    model_slug = "lstm_akf"
    layout = _prepare_model_layout(root_dir, model_slug)
    checkpoint_path = layout["checkpoints"] / "best.pt"
    config_payload = {
        "model_name": model_name,
        "model_type": model_slug,
        "trainable": False,
        "checkpoint_mode": "external_only",
        "expected_checkpoint": str(checkpoint_path),
        "dataset": dataset_config,
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(layout["train"] / "config.json", config_payload)

    if not checkpoint_path.exists():
        reason = f"Missing external checkpoint: {checkpoint_path}"
        write_json(
            layout["train"] / "summary.json",
            {
                "status": "waiting_for_external_checkpoint",
                "reason": reason,
                "expected_checkpoint": str(checkpoint_path),
            },
        )
        print(f"[{model_name}] skipped: {reason}", flush=True)
        return {
            "model_name": model_name,
            "model_slug": model_slug,
            "layout": layout,
            "skipped": True,
            "reason": reason,
        }

    try:
        model, loaded_config = _load_trained_torch_model(checkpoint_path, device)
    except Exception as exc:
        reason = f"External checkpoint is incompatible or unreadable: {exc}"
        write_json(
            layout["train"] / "summary.json",
            {
                "status": "skipped",
                "reason": reason,
                "expected_checkpoint": str(checkpoint_path),
            },
        )
        print(f"[{model_name}] skipped: {reason}", flush=True)
        return {
            "model_name": model_name,
            "model_slug": model_slug,
            "layout": layout,
            "skipped": True,
            "reason": reason,
        }

    if not loaded_config.use_baseline:
        reason = f"External checkpoint at {checkpoint_path} is not an LSTM-AKF model."
        write_json(
            layout["train"] / "summary.json",
            {
                "status": "skipped",
                "reason": reason,
                "expected_checkpoint": str(checkpoint_path),
            },
        )
        print(f"[{model_name}] skipped: {reason}", flush=True)
        return {
            "model_name": model_name,
            "model_slug": model_slug,
            "layout": layout,
            "skipped": True,
            "reason": reason,
        }

    total_params, trainable_params = _count_parameters(model)
    write_json(
        layout["train"] / "summary.json",
        {
            "status": "loaded_external_checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
    )
    print(f"[{model_name}] loaded external checkpoint: {checkpoint_path}", flush=True)
    return {
        "model": model,
        "layout": layout,
        "artifact_path": checkpoint_path,
        "model_name": model_name,
        "model_slug": model_slug,
        "use_baseline": loaded_config.use_baseline,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _train_or_reuse_torch_model(
    model_name: str,
    model_slug: str,
    use_baseline: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sample_count: int,
    val_sample_count: int,
    device: torch.device,
    root_dir: Path,
    dataset_fingerprint: dict[str, Any],
    batch_size: int,
    epochs: int,
    dataset_config: dict[str, float | int | bool],
    no_progress: bool,
    no_plot: bool,
) -> dict[str, Any]:
    layout = _prepare_model_layout(root_dir, model_slug)
    checkpoint_path = layout["checkpoints"] / "best.pt"
    model_config = _build_model_config(use_baseline=use_baseline)
    residual_aux_weight = DEFAULT_RESIDUAL_AUX_WEIGHT if use_baseline else 0.0
    direct_aux_weight = DEFAULT_DIRECT_AUX_WEIGHT if use_baseline else 0.0
    config_payload = {
        "model_name": model_name,
        "model_type": model_slug,
        "trainable": True,
        "model": asdict(model_config),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "lr": 1e-3,
        "scheduler_step_size": 30,
        "scheduler_gamma": 0.5,
        "weight_decay": 0.0,
        "beta": 1.0,
        "residual_aux_weight": residual_aux_weight,
        "direct_aux_weight": direct_aux_weight,
        "device": str(device),
        "dataset": dataset_config,
        "dataset_fingerprint": dataset_fingerprint,
    }

    config_path = layout["train"] / "config.json"
    existing_config = None
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))

    should_train = not checkpoint_path.exists()
    if checkpoint_path.exists():
        if existing_config != config_payload:
            should_train = True
            print(f"[{model_name}] dataset/config changed, retraining checkpoint.", flush=True)
        else:
            try:
                _load_trained_torch_model(checkpoint_path, device)
            except Exception as exc:
                should_train = True
                print(
                    f"[{model_name}] checkpoint is incompatible with the current model, retraining: {exc}",
                    flush=True,
                )
            else:
                print(f"[{model_name}] existing checkpoint found, skip training: {checkpoint_path}", flush=True)

    write_json(config_path, config_payload)

    if should_train:
        model = ArmorXYResidualPredictor(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        scheduler = _build_scheduler(
            argparse.Namespace(
                scheduler="step",
                scheduler_step_size=30,
                scheduler_gamma=0.5,
                scheduler_t_max=None,
                scheduler_patience=10,
                min_lr=1e-6,
                epochs=epochs,
            ),
            optimizer,
        )
        loss_fn = MultiStepSmoothL1Loss(beta=1.0)

        print(f"\n[{model_name}] training start", flush=True)
        print(f"[{model_name}] train samples: {train_sample_count}", flush=True)
        print(f"[{model_name}] val samples: {val_sample_count}", flush=True)

        best_score = float("inf")
        best_metrics: dict[str, float] | None = None
        history_rows: list[dict[str, float | int | str]] = []

        for epoch in range(1, epochs + 1):
            print(f"\n[{model_name}] Epoch {epoch}/{epochs}", flush=True)
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
                total_epochs=epochs,
                show_progress=not no_progress,
            )
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                residual_aux_weight=residual_aux_weight,
                direct_aux_weight=direct_aux_weight,
                epoch=epoch,
                total_epochs=epochs,
                show_progress=not no_progress,
            )
            monitored = val_metrics["loss"]
            log_row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ade": train_metrics["ade"],
                "train_fde": train_metrics["fde"],
                "val_loss": val_metrics["loss"],
                "val_ade": val_metrics["ade"],
                "val_fde": val_metrics["fde"],
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
                metrics=val_metrics,
                config=config_payload,
            )

            if monitored < best_score:
                best_score = monitored
                best_metrics = dict(val_metrics)
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=best_metrics,
                    config=config_payload,
                )

            if scheduler is not None:
                scheduler.step()

            if not no_plot:
                plot_training_curves(history_rows, layout["train"] / "curves.png", title=f"{model_name} Training Curves")

            print(
                f"[{model_name}] Train loss={train_metrics['loss']:.6f}, ade={train_metrics['ade']:.6f}, fde={train_metrics['fde']:.6f}",
                flush=True,
            )
            print(
                f"[{model_name}] Val   loss={val_metrics['loss']:.6f}, ade={val_metrics['ade']:.6f}, fde={val_metrics['fde']:.6f}",
                flush=True,
            )

        if best_metrics is not None:
            write_json(layout["train"] / "summary.json", {"best_score": best_score, **best_metrics})

    model, loaded_config = _load_trained_torch_model(checkpoint_path, device)
    total_params, trainable_params = _count_parameters(model)
    return {
        "model": model,
        "layout": layout,
        "artifact_path": checkpoint_path,
        "model_name": model_name,
        "model_slug": model_slug,
        "use_baseline": loaded_config.use_baseline,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _train_or_reuse_classical_model(
    model_name: str,
    model_slug: str,
    train_arrays: DatasetArrays,
    root_dir: Path,
    dataset_fingerprint: dict[str, Any],
    no_plot: bool,
) -> dict[str, Any] | None:
    layout = _prepare_model_layout(root_dir, model_slug)
    artifact_path = layout["artifacts"] / "model.pkl"
    config_payload = {
        "model_name": model_name,
        "model_type": model_slug,
        "trainable": True,
        "feature_shape": list(train_arrays.features.shape),
        "target_shape": list(train_arrays.target_flat.shape),
        "dataset_fingerprint": dataset_fingerprint,
    }
    config_path = layout["train"] / "config.json"
    existing_config = None
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    write_json(config_path, config_payload)

    if not _has_module("sklearn"):
        reason = "sklearn is not installed, classical baseline training is skipped."
        write_json(
            layout["train"] / "summary.json",
            {
                "status": "skipped",
                "reason": reason,
                "required_dependency": "scikit-learn",
            },
        )
        print(f"[{model_name}] skipped: {reason}", flush=True)
        return {
            "model_name": model_name,
            "model_slug": model_slug,
            "layout": layout,
            "skipped": True,
            "reason": reason,
        }

    if artifact_path.exists() and existing_config == config_payload:
        print(f"[{model_name}] existing artifact found, skip training: {artifact_path}", flush=True)
        estimator = _load_pickle(artifact_path)
    else:
        if artifact_path.exists():
            print(f"[{model_name}] dataset/config changed, refitting artifact: {artifact_path}", flush=True)
        estimator = _fit_classical_estimator(model_slug)
        print(f"\n[{model_name}] fitting classical estimator", flush=True)
        start_time = time.perf_counter()
        estimator.fit(train_arrays.features, train_arrays.target_flat)
        fit_seconds = time.perf_counter() - start_time
        _save_pickle(artifact_path, estimator)

        summary_payload = {
            "fit_seconds": fit_seconds,
            "train_samples": int(train_arrays.features.shape[0]),
            "feature_dim": int(train_arrays.features.shape[1]),
            "target_dim": int(train_arrays.target_flat.shape[1]),
            "artifact_path": str(artifact_path),
            "estimator_params": _serialize_estimator_params(estimator),
        }
        write_json(layout["train"] / "summary.json", summary_payload)
        _append_log(
            layout["train"] / "log.csv",
            {
                "step": "fit",
                "fit_seconds": fit_seconds,
                "train_samples": int(train_arrays.features.shape[0]),
            },
        )

        loss_curve: list[float] | None = None
        if model_slug == "ann":
            ann_model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
            loss_curve = [float(value) for value in getattr(ann_model, "loss_curve_", [])]
        if loss_curve:
            history_rows = _build_training_history_from_loss_curve(loss_curve)
            for row in history_rows:
                _append_log(layout["train"] / "loss_curve.csv", row)
            if not no_plot:
                plot_training_curves(history_rows, layout["train"] / "curves.png", title=f"{model_name} Training Curves")

    return {
        "estimator": estimator,
        "layout": layout,
        "artifact_path": artifact_path,
        "model_name": model_name,
        "model_slug": model_slug,
        "use_baseline": False,
        "total_params": 0,
        "trainable_params": 0,
    }


def _evaluate_prediction_set(
    predictions: np.ndarray,
    targets: np.ndarray,
    inference_seconds: list[float],
    point_frames: list[int],
    sequence_horizons: list[int],
    hit_threshold: float,
) -> dict[str, Any]:
    pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    inference_ms = [float(value * 1000.0) for value in inference_seconds]

    scopes: dict[str, Any] = {}
    for frame in point_frames:
        metric_arrays = point_metric_arrays(
            pred_tensor,
            target_tensor,
            frame - 1,
            hit_threshold=hit_threshold,
        )
        metric_arrays["inference_time"] = inference_ms
        scopes[f"point_f{frame:02d}"] = summarize_metric_arrays(metric_arrays)
    for horizon in sequence_horizons:
        metric_arrays = sequence_metric_arrays(
            pred_tensor,
            target_tensor,
            horizon,
            hit_threshold=hit_threshold,
        )
        metric_arrays["inference_time"] = inference_ms
        scopes[f"sequence_h{horizon:02d}"] = summarize_metric_arrays(metric_arrays)
    return scopes


def _run_model_predictions(
    model_info: dict[str, Any],
    val_arrays: DatasetArrays,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[float]]:
    model_slug = model_info["model_slug"]
    if model_slug == "akf_only":
        start_time = time.perf_counter()
        predictions = val_arrays.baseline_xy.copy()
        elapsed = time.perf_counter() - start_time
        per_sample = elapsed / max(1, len(predictions))
        return predictions, [per_sample] * len(predictions)
    if "model" in model_info:
        return _predict_torch_model(model_info["model"], val_arrays, device, batch_size)
    return _predict_classical_model(
        model_info["estimator"],
        val_arrays.features,
        future_steps=val_arrays.target_xy.shape[1],
    )


def _flatten_scope_summary(model_name: str, scope_name: str, scope_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "scope": scope_name,
        "de": float(scope_payload["de"]["mean"]),
        "mse": float(scope_payload["mse"]["mean"]),
        "rmse": float(scope_payload["rmse"]["mean"]),
        "mae": float(scope_payload["mae"]["mean"]),
        "hit_rate": float(scope_payload["hit_rate"]["mean"]),
        "inference_time_ms": float(scope_payload["inference_time"]["mean"]),
    }


def _write_comparison_csv(path: Path, detailed_results: dict[str, Any]) -> Path:
    fieldnames = [
        "scope",
        "model_name",
        "de",
        "mse",
        "rmse",
        "mae",
        "hit_rate",
        "inference_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scope_name, scope_payload in detailed_results["comparison"].items():
            for model_name, model_payload in scope_payload.items():
                writer.writerow(_flatten_scope_summary(model_name, scope_name, model_payload))
    return path


def _plot_scope_results(
    comparison_root: Path,
    scope_name: str,
    scope_payload: dict[str, Any],
    hit_threshold: float,
) -> Path | None:
    rows = []
    for model_name, metrics in scope_payload.items():
        rows.append(_flatten_scope_summary(model_name, scope_name, metrics))
    rows = sorted(rows, key=lambda row: (row["de"], row["rmse"], row["mae"]))
    metric_specs = [
        ("de", "DE"),
        ("mse", "MSE"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("hit_rate", f"Hit Rate@{hit_threshold:g}"),
        ("inference_time_ms", "Inference Time (ms)"),
    ]
    title = f"Model Comparison - {scope_name}"
    return plot_metric_grid(
        rows,
        metric_specs=metric_specs,
        output_path=comparison_root / f"{scope_name}_metrics.png",
        title=title,
    )


def _scope_title(scope_name: str) -> str:
    if scope_name.startswith("point_f"):
        frame = int(scope_name.split("f")[-1])
        return f"单帧第 {frame} 帧对比"
    if scope_name.startswith("sequence_h"):
        horizon = int(scope_name.split("h")[-1])
        return f"连续前 {horizon} 帧对比"
    return scope_name


def _scope_sort_rule(scope_name: str) -> str:
    if scope_name.startswith("point_f"):
        return "按 DE -> RMSE -> MAE 升序排序。"
    return "按平均 DE -> RMSE -> MAE 升序排序。"


def _write_scope_summary_markdown(
    path: Path,
    scope_name: str,
    rows: list[dict[str, Any]],
    skipped_models: dict[str, Any],
    hit_threshold: float,
) -> Path:
    title = _scope_title(scope_name)
    best_model = rows[0]["model_name"] if rows else "N/A"
    lines = [
        f"# {title}",
        "",
        f"- 最佳模型：`{best_model}`",
        f"- 排序规则：{_scope_sort_rule(scope_name)}",
        f"- 命中阈值：`DE <= {hit_threshold:g}` 记为命中。",
        "- 指标方向：`DE / MSE / RMSE / MAE / Inference Time` 越低越好，`Hit Rate` 越高越好。",
        "",
        "| 排名 | 模型 | DE | MSE | RMSE | MAE | Hit Rate | Inference Time (ms) |",
        "|------|------|----|-----|------|-----|----|----------------------|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['model_name']} | "
            f"{row['de']:.6f} | {row['mse']:.6f} | {row['rmse']:.6f} | "
            f"{row['mae']:.6f} | {row['hit_rate']:.6f} | {row['inference_time_ms']:.6f} |"
        )
    if skipped_models:
        lines.extend(
            [
                "",
                "## 跳过模型",
                "",
            ]
        )
        for item in skipped_models.values():
            lines.append(f"- `{item['model_name']}`：{item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_overall_summary(scope_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    scope_count = len(scope_rows)
    for scope_name, rows in scope_rows.items():
        for rank, row in enumerate(rows, start=1):
            item = aggregate.setdefault(
                row["model_name"],
                {
                    "model_name": row["model_name"],
                    "rank_sum": 0,
                    "wins": 0,
                    "avg_de": 0.0,
                    "avg_rmse": 0.0,
                    "avg_mae": 0.0,
                    "scope_ranks": {},
                },
            )
            item["rank_sum"] += rank
            item["wins"] += 1 if rank == 1 else 0
            item["avg_de"] += float(row["de"])
            item["avg_rmse"] += float(row["rmse"])
            item["avg_mae"] += float(row["mae"])
            item["scope_ranks"][scope_name] = rank
    overall_rows = list(aggregate.values())
    for row in overall_rows:
        row["rank_mean"] = row["rank_sum"] / max(1, scope_count)
        row["avg_de"] /= max(1, scope_count)
        row["avg_rmse"] /= max(1, scope_count)
        row["avg_mae"] /= max(1, scope_count)
    overall_rows.sort(key=lambda row: (row["rank_sum"], -row["wins"], row["avg_de"], row["avg_rmse"], row["avg_mae"]))
    return overall_rows


def _write_overall_summary_markdown(
    path: Path,
    overall_rows: list[dict[str, Any]],
    scope_rows: dict[str, list[dict[str, Any]]],
    skipped_models: dict[str, Any],
    hit_threshold: float,
) -> Path:
    scope_names = list(scope_rows.keys())
    best_model = overall_rows[0]["model_name"] if overall_rows else "N/A"
    header = "| 总排名 | 模型 | 排名总和 | 平均排名 | 冠军次数 | 平均 DE | " + " | ".join(scope_names) + " |"
    split = "|------|------|----------|----------|----------|---------|" + "|".join(["---"] * len(scope_names)) + "|"
    lines = [
        "# 总性能总结",
        "",
        f"- 最佳综合模型：`{best_model}`",
        "- 统计规则：把每个对比范围内的名次相加，排名总和越小越好；若并列，则冠军次数更多者更优。",
        f"- 命中阈值：`DE <= {hit_threshold:g}` 记为命中。",
        "- 指标方向：`DE / MSE / RMSE / MAE / Inference Time` 越低越好，`Hit Rate` 越高越好。",
        "",
        header,
        split,
    ]
    for overall_rank, row in enumerate(overall_rows, start=1):
        scope_rank_values = [str(row["scope_ranks"].get(scope_name, "-")) for scope_name in scope_names]
        lines.append(
            f"| {overall_rank} | {row['model_name']} | {row['rank_sum']} | {row['rank_mean']:.3f} | "
            f"{row['wins']} | {row['avg_de']:.6f} | " + " | ".join(scope_rank_values) + " |"
        )
    if skipped_models:
        lines.extend(
            [
                "",
                "## 跳过模型",
                "",
            ]
        )
        for item in skipped_models.values():
            lines.append(f"- `{item['model_name']}`：{item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train or reuse baseline models and compare them.")
    parser.add_argument("--train-data", type=Path, default=root / "data" / "dataset" / "train")
    parser.add_argument("--val-data", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--output-dir", type=Path, default=root / "Baseline Models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--point-frames", nargs="+", type=int, default=list(DEFAULT_POINT_FRAMES))
    parser.add_argument("--sequence-horizons", nargs="+", type=int, default=list(DEFAULT_SEQUENCE_HORIZONS))
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=DEFAULT_HIT_THRESHOLD,
        help="Normalized DE threshold used for hit rate metrics.",
    )
    _add_akf_args(parser)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate metric figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    comparison_layout = _prepare_comparison_layout(output_dir)
    device = torch.device(args.device)
    model_config = _build_model_config(use_baseline=True)
    dataset_kwargs = _build_dataset_kwargs(args, model_config)

    if _count_json_files(args.train_data) <= 0:
        raise FileNotFoundError(f"No training json files found under: {args.train_data}")
    if _count_json_files(args.val_data) <= 0:
        raise FileNotFoundError(f"No validation json files found under: {args.val_data}")

    point_frames = _normalize_indices(args.point_frames, model_config.future_steps, "point frame")
    sequence_horizons = _normalize_indices(args.sequence_horizons, model_config.future_steps, "sequence horizon")

    train_dataset, train_loader, val_dataset, val_loader = _load_dataset_objects(
        args.train_data,
        args.val_data,
        batch_size=DEFAULT_COMPARE_BATCH_SIZE,
        num_workers=0,
        dataset_kwargs=dataset_kwargs,
    )
    dataset_fingerprint = {
        "train": _dataset_fingerprint(train_dataset),
        "val": _dataset_fingerprint(val_dataset),
    }
    train_arrays = _dataset_to_arrays(train_dataset)
    val_arrays = _dataset_to_arrays(val_dataset)

    write_json(
        comparison_layout["root"] / "run_config.json",
        {
            "models": DEFAULT_MODEL_ORDER,
            "train_data": str(args.train_data),
            "val_data": str(args.val_data),
            "device": args.device,
            "point_frames": point_frames,
            "sequence_horizons": sequence_horizons,
            "hit_threshold": float(args.hit_threshold),
            "dataset": dataset_kwargs,
            "dataset_fingerprint": dataset_fingerprint,
            "lstm_akf_checkpoint_mode": "external_only",
            "lstm_akf_expected_checkpoint": str(output_dir / "lstm_akf" / "checkpoints" / "best.pt"),
            "fixed_training_config": {
                "epochs": DEFAULT_COMPARE_EPOCHS,
                "batch_size": DEFAULT_COMPARE_BATCH_SIZE,
                "optimizer": "Adam",
                "lr": 1e-3,
                "scheduler": "StepLR",
                "scheduler_step_size": 30,
                "scheduler_gamma": 0.5,
                "residual_aux_weight": DEFAULT_RESIDUAL_AUX_WEIGHT,
                "direct_aux_weight": DEFAULT_DIRECT_AUX_WEIGHT,
            },
        },
    )

    print(f"Baseline comparison root: {output_dir}", flush=True)
    print(f"Train data: {args.train_data}", flush=True)
    print(f"Val data: {args.val_data}", flush=True)
    print(f"Point frames: {point_frames}", flush=True)
    print(f"Sequence horizons: {sequence_horizons}", flush=True)
    print(f"Hit threshold: {args.hit_threshold}", flush=True)

    trained_models: list[dict[str, Any]] = []
    skipped_models: dict[str, Any] = {}
    lstm_akf_info = _load_external_lstm_akf_model(
        root_dir=output_dir,
        device=device,
        dataset_config=dataset_kwargs,
        dataset_fingerprint=dataset_fingerprint,
    )
    if lstm_akf_info.get("skipped"):
        skipped_models["lstm_akf"] = {
            "model_name": lstm_akf_info["model_name"],
            "reason": lstm_akf_info["reason"],
        }
    else:
        trained_models.append(lstm_akf_info)

    trained_models.append(
        _train_or_reuse_torch_model(
            model_name="LSTM-only",
            model_slug="lstm_only",
            use_baseline=False,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sample_count=len(train_dataset),
            val_sample_count=len(val_dataset),
            device=device,
            root_dir=output_dir,
            dataset_fingerprint=dataset_fingerprint,
            batch_size=DEFAULT_COMPARE_BATCH_SIZE,
            epochs=DEFAULT_COMPARE_EPOCHS,
            dataset_config=dataset_kwargs,
            no_progress=args.no_progress,
            no_plot=args.no_plot,
        )
    )

    akf_layout = _prepare_model_layout(output_dir, "akf_only")
    write_json(
        akf_layout["train"] / "config.json",
        {
            "model_name": "AKF-only",
            "model_type": "akf_only",
            "trainable": False,
            "future_steps": model_config.future_steps,
            "dataset": dataset_kwargs,
        },
    )
    trained_models.append(
        {
            "model_name": "AKF-only",
            "model_slug": "akf_only",
            "layout": akf_layout,
            "use_baseline": True,
            "total_params": 0,
            "trainable_params": 0,
        }
    )

    classical_specs = [
        ("ANN", "ann"),
        ("SVR", "svr"),
        ("AR", "ar"),
        ("DT", "dt"),
        ("KNN", "knn"),
    ]
    for model_name, model_slug in classical_specs:
        model_info = _train_or_reuse_classical_model(
            model_name=model_name,
            model_slug=model_slug,
            train_arrays=train_arrays,
            root_dir=output_dir,
            dataset_fingerprint=dataset_fingerprint,
            no_plot=args.no_plot,
        )
        if model_info is None:
            continue
        if model_info.get("skipped"):
            skipped_models[model_slug] = {
                "model_name": model_name,
                "reason": model_info["reason"],
            }
            continue
        trained_models.append(model_info)

    detailed_results: dict[str, Any] = {
        "models": {},
        "skipped_models": skipped_models,
        "comparison": {f"point_f{frame:02d}": {} for frame in point_frames},
    }
    for horizon in sequence_horizons:
        detailed_results["comparison"][f"sequence_h{horizon:02d}"] = {}

    for model_info in trained_models:
        predictions, inference_values = _run_model_predictions(
            model_info,
            val_arrays,
            device,
            batch_size=DEFAULT_COMPARE_BATCH_SIZE,
        )
        model_results = _evaluate_prediction_set(
            predictions=predictions,
            targets=val_arrays.target_xy,
            inference_seconds=inference_values,
            point_frames=point_frames,
            sequence_horizons=sequence_horizons,
            hit_threshold=float(args.hit_threshold),
        )
        detailed_results["models"][model_info["model_slug"]] = {
            "model_name": model_info["model_name"],
            "model_slug": model_info["model_slug"],
            "use_baseline": model_info["use_baseline"],
            "artifact_path": str(model_info.get("artifact_path") or ""),
            "total_params": int(model_info["total_params"]),
            "trainable_params": int(model_info["trainable_params"]),
            "results": model_results,
        }
        write_json(model_info["layout"]["val"] / "metrics.json", model_results)
        for scope_name, scope_payload in model_results.items():
            detailed_results["comparison"][scope_name][model_info["model_name"]] = scope_payload

    summary_results: dict[str, Any] = {}
    ranked_scope_rows: dict[str, list[dict[str, Any]]] = {}
    for scope_name, scope_payload in detailed_results["comparison"].items():
        rows = []
        for model_name, metrics in scope_payload.items():
            rows.append(_flatten_scope_summary(model_name, scope_name, metrics))
        rows = sorted(rows, key=lambda row: (row["de"], row["rmse"], row["mae"]))
        ranked_scope_rows[scope_name] = rows
        scope_summary: dict[str, Any] = {}
        for rank, row in enumerate(rows, start=1):
            scope_summary[row["model_name"]] = {
                "rank": rank,
                "DE": row["de"],
                "MSE": row["mse"],
                "RMSE": row["rmse"],
                "MAE": row["mae"],
                "Hit_Rate": row["hit_rate"],
                "Inference_Time_ms": row["inference_time_ms"],
            }
        summary_results[scope_name] = scope_summary
        if not args.no_plot:
            _plot_scope_results(
                comparison_layout["root"],
                scope_name,
                scope_payload,
                hit_threshold=float(args.hit_threshold),
            )
        _write_scope_summary_markdown(
            comparison_layout["docs"] / f"{scope_name}_summary.md",
            scope_name=scope_name,
            rows=rows,
            skipped_models=skipped_models,
            hit_threshold=float(args.hit_threshold),
        )

    overall_rows = _build_overall_summary(ranked_scope_rows)
    summary_results["overall"] = {
        row["model_name"]: {
            "rank_sum": row["rank_sum"],
            "rank_mean": row["rank_mean"],
            "wins": row["wins"],
            "avg_DE": row["avg_de"],
            "scope_ranks": row["scope_ranks"],
        }
        for row in overall_rows
    }
    _write_overall_summary_markdown(
        comparison_layout["docs"] / "overall_summary.md",
        overall_rows=overall_rows,
        scope_rows=ranked_scope_rows,
        skipped_models=skipped_models,
        hit_threshold=float(args.hit_threshold),
    )

    write_json(comparison_layout["detailed"], detailed_results)
    write_json(comparison_layout["summary"], summary_results)
    _write_comparison_csv(comparison_layout["csv"], detailed_results)
    if skipped_models:
        write_json(comparison_layout["root"] / "skipped_models.json", skipped_models)

    print("\nComparison finished.", flush=True)
    print(f"Comparison outputs: {comparison_layout['root']}", flush=True)


__all__ = ["main", "parse_args"]
