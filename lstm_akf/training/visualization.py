from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from lstm_akf.datasets.builder import JsonSampleDataset
from lstm_akf.models import ArmorXYModelConfig, ArmorXYResidualPredictor
from lstm_akf.training.checkpoint import load_checkpoint


MODEL_ORDER = [
    "lstm_akf",
    "akf_only",
    "lstm_only",
    "ann",
    "svr",
    "ar",
    "dt",
    "knn",
]
MODEL_LABELS = {
    "lstm_akf": "LSTM-AKF",
    "akf_only": "AKF-only",
    "lstm_only": "LSTM-only",
    "ann": "ANN",
    "svr": "SVR",
    "ar": "AR",
    "dt": "DT",
    "knn": "KNN",
}
MODEL_COLORS = {
    "lstm_akf": (40, 40, 240),
    "akf_only": (0, 165, 255),
    "lstm_only": (235, 120, 40),
    "ann": (220, 70, 180),
    "svr": (180, 90, 20),
    "ar": (20, 190, 190),
    "dt": (130, 90, 210),
    "knn": (80, 210, 80),
}
TRUE_COLOR = (40, 220, 40)
HISTORY_COLOR = (180, 180, 180)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
SAMPLE_RE = re.compile(r"^(?P<video_stem>.+)_seg(?P<segment_index>\d+)_sam(?P<sample_index>\d+)\.json$")


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for visualization") from _CV2_IMPORT_ERROR


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_sample_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_dir():
        files = sorted(dataset_path.glob("*.json"))
    else:
        files = [dataset_path]
    if not files:
        raise FileNotFoundError(f"No JSON samples found under: {dataset_path}")
    return files


def _resolve_sample_paths(args: argparse.Namespace) -> list[Path]:
    if args.sample_file is not None:
        sample_path = args.sample_file
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        return [sample_path]

    files = _list_sample_files(args.dataset)
    if args.sample_index < 0 or args.sample_index >= len(files):
        raise IndexError(f"--sample-index must be in [0, {len(files) - 1}], got {args.sample_index}")
    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be greater than 0, got {args.num_samples}")
    if args.sample_stride <= 0:
        raise ValueError(f"--sample-stride must be greater than 0, got {args.sample_stride}")
    sample_paths = files[args.sample_index :: args.sample_stride]
    return sample_paths[: args.num_samples]


def _metadata_from_filename(sample_path: Path) -> dict[str, Any]:
    match = SAMPLE_RE.match(sample_path.name)
    if match is None:
        return {}
    data = match.groupdict()
    return {
        "video_stem": data["video_stem"],
        "segment_index": int(data["segment_index"]),
        "sample_index": int(data["sample_index"]),
    }


def _resolve_video_path(meta: dict[str, Any], videos_dir: Path) -> Path:
    video_path = meta.get("video_path")
    if video_path:
        candidate = Path(video_path)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            rooted = _project_root() / candidate
            if rooted.exists():
                return rooted

    video_name = meta.get("video_name")
    if video_name:
        candidate = videos_dir / str(video_name)
        if candidate.exists():
            return candidate

    video_stem = meta.get("video_stem")
    if video_stem:
        for suffix in VIDEO_EXTENSIONS:
            candidate = videos_dir / f"{video_stem}{suffix}"
            if candidate.exists():
                return candidate
        matches = sorted(videos_dir.rglob(f"{video_stem}.*")) if videos_dir.exists() else []
        for candidate in matches:
            if candidate.suffix.lower() in VIDEO_EXTENSIONS:
                return candidate

    raise FileNotFoundError(
        "Unable to resolve video file from sample metadata. "
        "Rebuild the dataset with metadata or pass --videos-dir containing the source videos."
    )


def _sequence_value(values: Any, index: int) -> Any:
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None
    return values[index]


def _frame_reference(meta: dict[str, Any], prefix: str, index: int) -> tuple[int | None, float | None]:
    frame_index = _sequence_value(meta.get(f"{prefix}_frame_indices"), index)
    timestamp = _sequence_value(meta.get(f"{prefix}_timestamps"), index)
    if timestamp is not None:
        timestamp = float(timestamp)
    if frame_index is not None:
        frame_index = int(frame_index)
    return frame_index, timestamp


def _read_video_frame(video_path: Path, frame_index: int | None = None, timestamp: float | None = None) -> np.ndarray:
    _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    try:
        if timestamp is not None and timestamp >= 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        elif frame_index is not None and frame_index >= 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to read frame from {video_path} at frame={frame_index}, timestamp={timestamp}")
    return frame


def _xy_to_px(point: Iterable[float], width: int, height: int) -> tuple[int, int]:
    x, y = [float(value) for value in point]
    px = int(round(x * max(width - 1, 1)))
    py = int(round(y * max(height - 1, 1)))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _points_to_px(points: np.ndarray, width: int, height: int) -> list[tuple[int, int]]:
    return [_xy_to_px(point, width, height) for point in points]


def _draw_marker(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    label: str | None = None,
    radius: int = 5,
) -> None:
    _require_cv2()
    cv2.circle(image, point, radius + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(image, point, radius, color, -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(image, point, color, markerType=cv2.MARKER_CROSS, markerSize=radius * 4, thickness=2)
    if label:
        cv2.putText(
            image,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_polyline(
    image: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
    closed: bool = False,
) -> None:
    _require_cv2()
    if len(points) >= 2:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], closed, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, max(2, thickness + 1), color, -1, lineType=cv2.LINE_AA)


def _draw_panel(image: np.ndarray, title: str, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    _require_cv2()
    line_height = 20
    width = 360
    height = 34 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.56, image, 0.44, 0, image)
    cv2.putText(image, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for index, (text, color) in enumerate(lines):
        y = 62 + index * line_height
        cv2.putText(image, text[:54], (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _crop_around_points(
    image: np.ndarray,
    points: list[tuple[int, int]],
    padding: int,
    min_size: int,
) -> np.ndarray:
    _require_cv2()
    if padding <= 0 or not points:
        return image
    height, width = image.shape[:2]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0, min(xs) - padding)
    right = min(width, max(xs) + padding)
    top = max(0, min(ys) - padding)
    bottom = min(height, max(ys) + padding)
    if right <= left or bottom <= top:
        return image
    cropped = image[top:bottom, left:right]
    scale = max(1.0, min_size / max(cropped.shape[0], cropped.shape[1], 1))
    if scale <= 1.01:
        return cropped
    return cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _extract_model_config(checkpoint: dict[str, Any]) -> ArmorXYModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(raw_config, dict) and isinstance(raw_config.get("model"), dict):
        raw_config = raw_config["model"]
    field_names = {field.name for field in fields(ArmorXYModelConfig)}
    payload = {key: raw_config[key] for key in field_names if key in raw_config}
    return ArmorXYModelConfig(**payload)


def _load_torch_model(checkpoint_path: Path, device: torch.device) -> ArmorXYResidualPredictor:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = _extract_model_config(checkpoint)
    model = ArmorXYResidualPredictor(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _predict_models(
    sample_path: Path,
    models_dir: Path,
    model_slugs: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = JsonSampleDataset(sample_path)
    item = dataset[0]
    history_xy = item["history_xy"].unsqueeze(0).to(device)
    baseline_xy = item["baseline_xy"].unsqueeze(0).to(device)
    history_len = item["history_len"].reshape(1).to(device)
    feature = item["history_xy"].detach().cpu().numpy().reshape(1, -1)

    predictions: dict[str, np.ndarray] = {}
    for model_slug in model_slugs:
        if model_slug == "akf_only":
            predictions[model_slug] = item["baseline_xy"].detach().cpu().numpy()
            continue

        if model_slug in {"lstm_akf", "lstm_only"}:
            checkpoint_path = models_dir / model_slug / "checkpoints" / "best.pt"
            if not checkpoint_path.exists():
                print(f"[visualize] skip {model_slug}: missing {checkpoint_path}", flush=True)
                continue
            model = _load_torch_model(checkpoint_path, device)
            with torch.no_grad():
                output = model(history_xy=history_xy, baseline_xy=baseline_xy, history_len=history_len)["pred_xy"]
            predictions[model_slug] = output[0].detach().cpu().numpy().astype(np.float32)
            continue

        artifact_path = models_dir / model_slug / "artifacts" / "model.pkl"
        if not artifact_path.exists():
            print(f"[visualize] skip {model_slug}: missing {artifact_path}", flush=True)
            continue
        estimator = _load_pickle(artifact_path)
        pred_flat = np.asarray(estimator.predict(feature), dtype=np.float32)
        predictions[model_slug] = pred_flat.reshape(-1, 2)

    return predictions


def _validate_horizons(horizons: list[int], future_steps: int) -> list[int]:
    normalized = sorted(set(int(value) for value in horizons))
    for value in normalized:
        if value <= 0 or value > future_steps:
            raise ValueError(f"--horizons values must be in [1, {future_steps}], got {value}")
    return normalized


def _write_image(path: Path, image: np.ndarray) -> Path:
    _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")
    return path


def _draw_single_source_panel(
    image: np.ndarray,
    source_label: str,
    color: tuple[int, int, int],
    detail: str,
) -> None:
    _draw_panel(image, source_label, [(detail, color)])


def _ordered_prediction_items(predictions: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    return [(model_slug, predictions[model_slug]) for model_slug in MODEL_ORDER if model_slug in predictions]


def _render_single_point_only(
    frame: np.ndarray,
    point_xy: np.ndarray,
    source_label: str,
    color: tuple[int, int, int],
    detail: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    image = frame.copy()
    height, width = image.shape[:2]
    point_px = _xy_to_px(point_xy, width, height)
    _draw_marker(image, point_px, color, label=source_label, radius=6)
    _draw_single_source_panel(image, source_label, color, detail)
    return image, point_px


def _render_single_trajectory_only(
    frame: np.ndarray,
    points_xy: np.ndarray,
    source_label: str,
    color: tuple[int, int, int],
    detail: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = frame.copy()
    height, width = image.shape[:2]
    points_px = _points_to_px(points_xy, width, height)
    _draw_polyline(image, points_px, color, thickness=3)
    if points_px:
        _draw_marker(image, points_px[-1], color, label=source_label, radius=6)
    _draw_single_source_panel(image, source_label, color, detail)
    return image, points_px


def _render_single_frame(
    frame: np.ndarray,
    horizon: int,
    history_xy: np.ndarray,
    target_xy: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = frame.copy()
    height, width = image.shape[:2]
    truth_px = _xy_to_px(target_xy[horizon - 1], width, height)
    history_px = _points_to_px(history_xy, width, height)
    pred_points: list[tuple[int, int]] = []

    _draw_polyline(image, history_px, HISTORY_COLOR, thickness=1)
    _draw_marker(image, truth_px, TRUE_COLOR, label="True", radius=6)

    rows: list[tuple[str, float, tuple[int, int, int]]] = []
    for model_slug, pred_xy in predictions.items():
        if horizon > len(pred_xy):
            continue
        color = MODEL_COLORS.get(model_slug, (255, 255, 255))
        label = MODEL_LABELS.get(model_slug, model_slug)
        pred_px = _xy_to_px(pred_xy[horizon - 1], width, height)
        pred_points.append(pred_px)
        cv2.line(image, truth_px, pred_px, color, 1, cv2.LINE_AA)
        _draw_marker(image, pred_px, color, label=label, radius=4)
        err_px = math.dist(truth_px, pred_px)
        rows.append((label, err_px, color))

    rows.sort(key=lambda row: row[1])
    panel_lines = [(f"{label}: {err_px:.1f}px", color) for label, err_px, color in rows]
    _draw_panel(image, f"Single frame h={horizon}", panel_lines)
    return image, [truth_px, *pred_points]


def _render_trajectory(
    frame: np.ndarray,
    history_xy: np.ndarray,
    target_xy: np.ndarray,
    predictions: dict[str, np.ndarray],
    steps: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    image = frame.copy()
    height, width = image.shape[:2]
    steps = min(steps, len(target_xy))
    history_px = _points_to_px(history_xy, width, height)
    target_px = _points_to_px(target_xy[:steps], width, height)
    crop_points = [*history_px, *target_px]

    _draw_polyline(image, history_px, HISTORY_COLOR, thickness=2)
    _draw_polyline(image, target_px, TRUE_COLOR, thickness=3)
    if target_px:
        _draw_marker(image, target_px[-1], TRUE_COLOR, label="True end", radius=6)

    rows: list[tuple[str, float, tuple[int, int, int]]] = []
    for model_slug, pred_xy in predictions.items():
        color = MODEL_COLORS.get(model_slug, (255, 255, 255))
        label = MODEL_LABELS.get(model_slug, model_slug)
        pred_steps = min(steps, len(pred_xy))
        if pred_steps <= 0:
            continue
        pred_px = _points_to_px(pred_xy[:pred_steps], width, height)
        crop_points.extend(pred_px)
        _draw_polyline(image, pred_px, color, thickness=2)
        _draw_marker(image, pred_px[-1], color, label=label, radius=4)
        true_for_pred = target_xy[:pred_steps]
        ade_px = np.linalg.norm((pred_xy[:pred_steps] - true_for_pred) * np.array([width - 1, height - 1]), axis=1)
        rows.append((label, float(np.mean(ade_px)), color))

    rows.sort(key=lambda row: row[1])
    panel_lines = [(f"{label}: ADE {ade_px:.1f}px", color) for label, ade_px, color in rows]
    _draw_panel(image, f"Trajectory steps={steps}", panel_lines)
    return image, crop_points


def _render_outputs(
    sample_path: Path,
    payload: dict[str, Any],
    meta: dict[str, Any],
    video_path: Path,
    predictions: dict[str, np.ndarray],
    horizons: list[int],
    output_dir: Path,
    crop_padding: int,
    crop_min_size: int,
    mode: str,
) -> list[Path]:
    history_xy = np.asarray(payload["input"], dtype=np.float32)
    target_xy = np.asarray(payload["target"], dtype=np.float32)
    stem = sample_path.stem
    outputs: list[Path] = []
    render_overlay = mode in {"overlay", "both"}
    render_separate = mode in {"separate", "both"}

    for horizon in horizons:
        frame_index, timestamp = _frame_reference(meta, "target", horizon - 1)
        if frame_index is None and timestamp is None:
            raise ValueError(
                f"Sample {sample_path} does not contain target frame metadata. "
                "Rebuild the dataset with scripts/build_dataset.py before visualizing on real frames."
            )
        frame = _read_video_frame(video_path, frame_index=frame_index, timestamp=timestamp)
        if render_overlay:
            image, crop_points = _render_single_frame(frame, horizon, history_xy, target_xy, predictions)
            output_path = output_dir / f"{stem}_single_h{horizon:02d}_overlay.png"
            outputs.append(_write_image(output_path, image))
            if crop_padding > 0:
                crop = _crop_around_points(image, crop_points, padding=crop_padding, min_size=crop_min_size)
                outputs.append(_write_image(output_dir / f"{stem}_single_h{horizon:02d}_overlay_crop.png", crop))

        if render_separate:
            height, width = frame.shape[:2]
            union_points = [_xy_to_px(target_xy[horizon - 1], width, height)]
            for _, pred_xy in _ordered_prediction_items(predictions):
                if horizon <= len(pred_xy):
                    union_points.append(_xy_to_px(pred_xy[horizon - 1], width, height))

            truth_image, _ = _render_single_point_only(
                frame,
                target_xy[horizon - 1],
                source_label="True",
                color=TRUE_COLOR,
                detail=f"h={horizon}",
            )
            truth_path = output_dir / f"{stem}_single_h{horizon:02d}_00_true.png"
            outputs.append(_write_image(truth_path, truth_image))
            if crop_padding > 0:
                crop = _crop_around_points(truth_image, union_points, padding=crop_padding, min_size=crop_min_size)
                outputs.append(_write_image(output_dir / f"{stem}_single_h{horizon:02d}_00_true_crop.png", crop))

            order_index = 1
            for model_slug, pred_xy in _ordered_prediction_items(predictions):
                if horizon > len(pred_xy):
                    continue
                label = MODEL_LABELS.get(model_slug, model_slug)
                color = MODEL_COLORS.get(model_slug, (255, 255, 255))
                pred_px = _xy_to_px(pred_xy[horizon - 1], width, height)
                truth_px = union_points[0]
                err_px = math.dist(truth_px, pred_px)
                model_image, _ = _render_single_point_only(
                    frame,
                    pred_xy[horizon - 1],
                    source_label=label,
                    color=color,
                    detail=f"h={horizon} | error={err_px:.1f}px",
                )
                model_path = output_dir / f"{stem}_single_h{horizon:02d}_{order_index:02d}_{model_slug}.png"
                outputs.append(_write_image(model_path, model_image))
                if crop_padding > 0:
                    crop = _crop_around_points(model_image, union_points, padding=crop_padding, min_size=crop_min_size)
                    outputs.append(
                        _write_image(
                            output_dir / f"{stem}_single_h{horizon:02d}_{order_index:02d}_{model_slug}_crop.png",
                            crop,
                        )
                    )
                order_index += 1

    frame_index, timestamp = _frame_reference(meta, "history", len(history_xy) - 1)
    if frame_index is None and timestamp is None:
        raise ValueError(
            f"Sample {sample_path} does not contain history frame metadata. "
            "Rebuild the dataset with scripts/build_dataset.py before visualizing on real frames."
        )
    frame = _read_video_frame(video_path, frame_index=frame_index, timestamp=timestamp)
    if render_overlay:
        image, crop_points = _render_trajectory(
            frame,
            history_xy=history_xy,
            target_xy=target_xy,
            predictions=predictions,
            steps=len(target_xy),
        )
        outputs.append(_write_image(output_dir / f"{stem}_trajectory_overlay.png", image))
        if crop_padding > 0:
            crop = _crop_around_points(image, crop_points, padding=crop_padding, min_size=crop_min_size)
            outputs.append(_write_image(output_dir / f"{stem}_trajectory_overlay_crop.png", crop))

    if render_separate:
        height, width = frame.shape[:2]
        trajectory_steps = len(target_xy)
        union_points = _points_to_px(target_xy[:trajectory_steps], width, height)
        for _, pred_xy in _ordered_prediction_items(predictions):
            pred_steps = min(trajectory_steps, len(pred_xy))
            union_points.extend(_points_to_px(pred_xy[:pred_steps], width, height))

        truth_image, _ = _render_single_trajectory_only(
            frame,
            target_xy[:trajectory_steps],
            source_label="True",
            color=TRUE_COLOR,
            detail=f"future steps={trajectory_steps}",
        )
        outputs.append(_write_image(output_dir / f"{stem}_trajectory_00_true.png", truth_image))
        if crop_padding > 0:
            crop = _crop_around_points(truth_image, union_points, padding=crop_padding, min_size=crop_min_size)
            outputs.append(_write_image(output_dir / f"{stem}_trajectory_00_true_crop.png", crop))

        order_index = 1
        for model_slug, pred_xy in _ordered_prediction_items(predictions):
            pred_steps = min(trajectory_steps, len(pred_xy))
            if pred_steps <= 0:
                continue
            label = MODEL_LABELS.get(model_slug, model_slug)
            color = MODEL_COLORS.get(model_slug, (255, 255, 255))
            ade_px = np.linalg.norm(
                (pred_xy[:pred_steps] - target_xy[:pred_steps]) * np.array([width - 1, height - 1]),
                axis=1,
            )
            model_image, _ = _render_single_trajectory_only(
                frame,
                pred_xy[:pred_steps],
                source_label=label,
                color=color,
                detail=f"future steps={pred_steps} | ADE={float(np.mean(ade_px)):.1f}px",
            )
            model_path = output_dir / f"{stem}_trajectory_{order_index:02d}_{model_slug}.png"
            outputs.append(_write_image(model_path, model_image))
            if crop_padding > 0:
                crop = _crop_around_points(model_image, union_points, padding=crop_padding, min_size=crop_min_size)
                outputs.append(_write_image(output_dir / f"{stem}_trajectory_{order_index:02d}_{model_slug}_crop.png", crop))
            order_index += 1

    return outputs


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Visualize model predictions on real video frames.")
    parser.add_argument("--dataset", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to render from --sample-index.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Step between rendered samples.")
    parser.add_argument("--sample-file", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=root / "Baseline Models")
    parser.add_argument("--videos-dir", type=Path, default=root / "data" / "video")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "visualizations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=MODEL_ORDER)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 8, 15])
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--crop-min-size", type=int, default=720)
    parser.add_argument(
        "--mode",
        choices=["separate", "overlay", "both"],
        default="separate",
        help="separate writes one source per image; overlay writes all sources on one image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sample_paths = _resolve_sample_paths(args)
    print(f"Rendering samples: {len(sample_paths)}", flush=True)
    for sample_path in sample_paths:
        payload = _load_json(sample_path)
        meta = {
            **_metadata_from_filename(sample_path),
            **dict(payload.get("meta") or {}),
        }
        video_path = _resolve_video_path(meta, args.videos_dir)
        target_xy = np.asarray(payload["target"], dtype=np.float32)
        horizons = _validate_horizons(args.horizons, future_steps=len(target_xy))
        predictions = _predict_models(sample_path, args.models_dir, list(args.models), device=device)
        if not predictions:
            raise RuntimeError(f"No model predictions were loaded from: {args.models_dir}")

        outputs = _render_outputs(
            sample_path=sample_path,
            payload=payload,
            meta=meta,
            video_path=video_path,
            predictions=predictions,
            horizons=horizons,
            output_dir=args.output_dir,
            crop_padding=args.crop_padding,
            crop_min_size=args.crop_min_size,
            mode=args.mode,
        )
        print(f"Sample: {sample_path}", flush=True)
        print(f"Video: {video_path}", flush=True)
        for output_path in outputs:
            print(f"Wrote: {output_path}", flush=True)


__all__ = ["main", "parse_args"]
