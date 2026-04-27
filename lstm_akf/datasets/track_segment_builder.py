from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from lstm_akf.models.akf import StateEstimatorConfig, XYStateEstimator


@dataclass
class TrackBuildConfig:
    max_missing: int = 5
    min_segment_length: int = 5
    dt: float = 1.0
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    initial_covariance: float = 1.0
    adaptive_measurement_noise: bool = True
    measurement_noise_min_scale: float = 0.25
    measurement_noise_max_scale: float = 16.0
    innovation_gain: float = 1.0
    innovation_smoothing: float = 0.2
    confidence_gain: float = 1.0
    missing_process_noise_scale: float = 1.35
    max_process_noise_scale: float = 6.0


def _build_estimator_config(config: TrackBuildConfig) -> StateEstimatorConfig:
    return StateEstimatorConfig(
        dt=config.dt,
        max_missing=config.max_missing,
        process_noise=config.process_noise,
        measurement_noise=config.measurement_noise,
        initial_covariance=config.initial_covariance,
        adaptive_measurement_noise=config.adaptive_measurement_noise,
        measurement_noise_min_scale=config.measurement_noise_min_scale,
        measurement_noise_max_scale=config.measurement_noise_max_scale,
        innovation_gain=config.innovation_gain,
        innovation_smoothing=config.innovation_smoothing,
        confidence_gain=config.confidence_gain,
        missing_process_noise_scale=config.missing_process_noise_scale,
        max_process_noise_scale=config.max_process_noise_scale,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_detection_rows(input_path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for raw in reader:
            cx = _safe_float(raw.get("cx"))
            cy = _safe_float(raw.get("cy"))
            has_detection = raw.get("has_detection")
            observed_flag = raw.get("observed")
            if observed_flag not in (None, ""):
                observed = str(observed_flag).strip().lower() in {"1", "true", "yes"}
            elif has_detection not in (None, ""):
                observed = bool(int(has_detection))
            else:
                observed = cx is not None and cy is not None
            rows.append(
                {
                    "video_name": raw.get("video_name") or raw.get("video_stem") or "unknown",
                    "video_stem": raw.get("video_stem") or Path(raw.get("video_name") or "unknown").stem,
                    "frame_index": _safe_int(raw.get("frame_index")) or 0,
                    "timestamp": _safe_float(raw.get("timestamp")),
                    "fps": _safe_float(raw.get("fps")) or 0.0,
                    "cx": cx,
                    "cy": cy,
                    "conf": _safe_float(raw.get("conf")),
                    "observed": observed and cx is not None and cy is not None,
                }
            )
    rows.sort(key=lambda item: (str(item["video_name"]), int(item["frame_index"])))
    return rows


def group_rows_by_video(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["video_name"]), []).append(dict(row))
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: int(item["frame_index"]))
    return grouped


def _append_point(
    segment: Dict[str, Any],
    frame_index: int,
    timestamp: Optional[float],
    xy: Sequence[float],
    observed: bool,
    conf: Optional[float],
) -> None:
    segment["points"].append(
        {
            "frame_index": int(frame_index),
            "timestamp": None if timestamp is None else float(timestamp),
            "cx": float(xy[0]),
            "cy": float(xy[1]),
            "observed": bool(observed),
            "conf": None if conf is None else float(conf),
        }
    )


def _new_segment(video_name: str, segment_index: int, fps: float) -> Dict[str, Any]:
    return {
        "video_name": video_name,
        "video_stem": Path(video_name).stem,
        "segment_index": int(segment_index),
        "fps": float(fps) if fps else 0.0,
        "points": [],
    }


def _finalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    points = segment["points"]
    if not points:
        segment["length"] = 0
        return segment
    segment["length"] = len(points)
    segment["start_frame_index"] = points[0]["frame_index"]
    segment["end_frame_index"] = points[-1]["frame_index"]
    segment["start_timestamp"] = points[0]["timestamp"]
    segment["end_timestamp"] = points[-1]["timestamp"]
    return segment


def build_track_segments(rows: Sequence[Dict[str, Any]], config: TrackBuildConfig) -> List[Dict[str, Any]]:
    grouped = group_rows_by_video(rows)
    segments: List[Dict[str, Any]] = []
    estimator_config = _build_estimator_config(config)

    for video_name, video_rows in grouped.items():
        video_rows = sorted(video_rows, key=lambda item: int(item["frame_index"]))
        fps = next((float(row["fps"]) for row in video_rows if row.get("fps")), 0.0) or 0.0
        estimator = XYStateEstimator(estimator_config)
        current_segment: Optional[Dict[str, Any]] = None
        segment_index = 0
        previous_frame_index: Optional[int] = None
        previous_timestamp: Optional[float] = None

        def flush_segment() -> None:
            nonlocal current_segment, segment_index
            if current_segment is None:
                return
            _finalize_segment(current_segment)
            if current_segment["length"] >= config.min_segment_length:
                segments.append(current_segment)
                segment_index += 1
            current_segment = None

        for row in video_rows:
            frame_index = int(row["frame_index"])
            timestamp = row.get("timestamp")
            row_fps = float(row.get("fps") or fps or 0.0)
            if row_fps > 0:
                fps = row_fps

            if previous_frame_index is not None and frame_index > previous_frame_index + 1:
                for missed_frame in range(previous_frame_index + 1, frame_index):
                    miss_timestamp = None
                    if fps > 0 and previous_timestamp is not None:
                        miss_timestamp = previous_timestamp + ((missed_frame - previous_frame_index) / fps)
                    result = estimator.update(None)
                    if current_segment is not None and result["active"] and result["xy"] is not None:
                        _append_point(
                            current_segment,
                            frame_index=missed_frame,
                            timestamp=miss_timestamp,
                            xy=result["xy"],
                            observed=False,
                            conf=None,
                        )
                    else:
                        flush_segment()

            observation = None
            has_detection = row.get("has_detection")
            row_observed = row.get("observed")
            if row_observed or (row_observed is None and has_detection):
                observation = (float(row["cx"]), float(row["cy"]))
            result = estimator.update(observation, confidence=row.get("conf"))

            if observation is not None:
                if current_segment is None:
                    current_segment = _new_segment(video_name, segment_index, fps)
                _append_point(
                    current_segment,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    xy=result["xy"],
                    observed=True,
                    conf=row.get("conf"),
                )
            elif current_segment is not None and result["active"] and result["xy"] is not None:
                _append_point(
                    current_segment,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    xy=result["xy"],
                    observed=False,
                    conf=None,
                )
            else:
                flush_segment()

            previous_frame_index = frame_index
            previous_timestamp = timestamp

        flush_segment()
    return segments


def save_track_segments(
    segments: Sequence[Dict[str, Any]],
    output_path: str | Path,
    config: TrackBuildConfig,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "segments": list(segments),
    }
    torch.save(payload, output_path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build continuous track segments from detection CSV rows.")
    parser.add_argument("--input-csv", type=Path, default=root / "data" / "detections.csv")
    parser.add_argument("--output-path", type=Path, default=root / "data" / "track_segments.pt")
    parser.add_argument("--max-missing", type=int, default=5)
    parser.add_argument("--min-segment-length", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--process-noise", type=float, default=1e-4)
    parser.add_argument("--measurement-noise", type=float, default=1e-3)
    parser.add_argument("--initial-covariance", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--fixed-measurement-noise",
        dest="adaptive_measurement_noise",
        action="store_false",
        help="Disable weak AKF adaptive measurement noise updates.",
    )
    parser.add_argument("--measurement-noise-min-scale", type=float, default=0.25)
    parser.add_argument("--measurement-noise-max-scale", type=float, default=16.0)
    parser.add_argument("--innovation-gain", type=float, default=1.0)
    parser.add_argument("--innovation-smoothing", type=float, default=0.2)
    parser.add_argument("--confidence-gain", type=float, default=1.0)
    parser.add_argument("--missing-process-noise-scale", type=float, default=1.35)
    parser.add_argument("--max-process-noise-scale", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_detection_rows(args.input_csv)
    config = TrackBuildConfig(
        max_missing=args.max_missing,
        min_segment_length=args.min_segment_length,
        dt=args.dt,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        initial_covariance=args.initial_covariance,
        adaptive_measurement_noise=args.adaptive_measurement_noise,
        measurement_noise_min_scale=args.measurement_noise_min_scale,
        measurement_noise_max_scale=args.measurement_noise_max_scale,
        innovation_gain=args.innovation_gain,
        innovation_smoothing=args.innovation_smoothing,
        confidence_gain=args.confidence_gain,
        missing_process_noise_scale=args.missing_process_noise_scale,
        max_process_noise_scale=args.max_process_noise_scale,
    )
    segments = build_track_segments(rows, config)
    save_track_segments(segments, args.output_path, config)


__all__ = [
    "TrackBuildConfig",
    "build_track_segments",
    "group_rows_by_video",
    "load_detection_rows",
    "main",
    "parse_args",
    "save_track_segments",
]
