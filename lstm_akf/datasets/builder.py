from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lstm_akf.datasets.detection_exporter import collect_detections
from lstm_akf.datasets.track_segment_builder import TrackBuildConfig, build_track_segments
from lstm_akf.models.akf import StateEstimatorConfig, XYStateEstimator


@dataclass
class SampleBuildConfig:
    min_history: int = 5
    max_history: int = 15
    future_steps: int = 15
    target_fps: float = 40.0
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


SAMPLE_FILENAME_PATTERN = re.compile(
    r"^(?P<video_stem>.+)_seg(?P<segment_index>\d+)_sam(?P<sample_index>\d+)\.json$"
)


def _non_none_items(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _resolve_sample_config(
    base_config: Dict[str, Any] | None = None,
    **config_overrides: Any,
) -> Dict[str, Any]:
    resolved = asdict(SampleBuildConfig())
    if isinstance(base_config, dict):
        resolved.update(_non_none_items(base_config))
    resolved.update(_non_none_items(config_overrides))
    return resolved


def _build_estimator_config(
    config: SampleBuildConfig | Dict[str, Any],
    *,
    max_missing: int = 5,
) -> StateEstimatorConfig:
    resolved = _resolve_sample_config(config if isinstance(config, dict) else asdict(config))

    return StateEstimatorConfig(
        dt=float(resolved["dt"]),
        max_missing=max_missing,
        process_noise=float(resolved["process_noise"]),
        measurement_noise=float(resolved["measurement_noise"]),
        initial_covariance=float(resolved["initial_covariance"]),
        adaptive_measurement_noise=bool(resolved["adaptive_measurement_noise"]),
        measurement_noise_min_scale=float(resolved["measurement_noise_min_scale"]),
        measurement_noise_max_scale=float(resolved["measurement_noise_max_scale"]),
        innovation_gain=float(resolved["innovation_gain"]),
        innovation_smoothing=float(resolved["innovation_smoothing"]),
        confidence_gain=float(resolved["confidence_gain"]),
        missing_process_noise_scale=float(resolved["missing_process_noise_scale"]),
        max_process_noise_scale=float(resolved["max_process_noise_scale"]),
    )


class AKFBaselineGenerator:
    def __init__(self, estimator_config: Optional[StateEstimatorConfig] = None, **config_kwargs: Any) -> None:
        self.estimator_config = estimator_config or StateEstimatorConfig(**config_kwargs)

    def predict_future(self, history_xy: torch.Tensor, future_steps: int) -> torch.Tensor:
        estimator = XYStateEstimator(self.estimator_config)
        for point in history_xy:
            estimator.update(point.tolist())
        future = estimator.predict_future(future_steps)
        return torch.as_tensor(future, dtype=torch.float32)


# Backward-compatible alias for older imports.
KalmanBaselineGenerator = AKFBaselineGenerator


def _to_float_tensor(xy: Sequence[Sequence[float]]) -> torch.Tensor:
    return torch.as_tensor(xy, dtype=torch.float32)


def _left_pad_history(history_xy: torch.Tensor, max_history: int) -> Tuple[torch.Tensor, int]:
    history_len = int(history_xy.shape[0])
    if history_len > max_history:
        history_xy = history_xy[-max_history:]
        history_len = max_history
    padded = torch.zeros((max_history, 2), dtype=torch.float32)
    padded[-history_len:] = history_xy
    return padded, history_len


def _sanitize_token(token: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in token.strip())
    return cleaned or "sample"


def _format_xy_sequence(xy: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(point[0]), float(point[1])] for point in xy]


def _parse_record_filename(filename: str) -> Dict[str, int | str]:
    match = SAMPLE_FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Unsupported sample filename format: {filename}")
    payload = match.groupdict()
    return {
        "video_stem": payload["video_stem"],
        "segment_index": int(payload["segment_index"]),
        "sample_index": int(payload["sample_index"]),
    }


def _record_group_key(
    record: Tuple[str, Dict[str, List[List[float]]]],
    split_mode: str,
) -> Tuple[str, ...]:
    filename, _ = record
    if split_mode == "sample":
        return ("sample", filename)

    parsed = _parse_record_filename(filename)
    if split_mode == "segment":
        return ("segment", str(parsed["video_stem"]), str(parsed["segment_index"]))
    if split_mode == "video":
        return ("video", str(parsed["video_stem"]))
    raise ValueError(f"Unsupported split mode: {split_mode}")


def _group_json_records(
    records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    split_mode: str,
) -> Dict[Tuple[str, ...], List[Tuple[str, Dict[str, List[List[float]]]]]]:
    grouped: Dict[Tuple[str, ...], List[Tuple[str, Dict[str, List[List[float]]]]]] = {}
    for record in records:
        grouped.setdefault(_record_group_key(record, split_mode), []).append(record)
    return grouped


def _serialize_group_key(group_key: Tuple[str, ...]) -> Dict[str, str | int]:
    if group_key[0] == "sample":
        parsed = _parse_record_filename(group_key[1])
        return {
            "split_mode": "sample",
            "video_stem": str(parsed["video_stem"]),
            "segment_index": int(parsed["segment_index"]),
            "sample_index": int(parsed["sample_index"]),
            "sample_filename": group_key[1],
        }
    if group_key[0] == "segment":
        return {
            "split_mode": "segment",
            "video_stem": group_key[1],
            "segment_index": int(group_key[2]),
        }
    if group_key[0] == "video":
        return {
            "split_mode": "video",
            "video_stem": group_key[1],
        }
    raise ValueError(f"Unsupported group key: {group_key}")


def _write_split_manifest(manifest: Dict[str, Any], manifest_path: str | Path | None) -> Optional[Path]:
    if manifest_path is None:
        return None
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _prepare_json_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()
    return output_dir


def _resample_with_timestamps(points: Sequence[Dict[str, Any]], target_fps: float) -> List[Dict[str, Any]]:
    if len(points) < 2 or target_fps <= 0:
        return [dict(point) for point in points]

    timestamps = [point.get("timestamp") for point in points]
    if any(timestamp is None for timestamp in timestamps):
        return [dict(point) for point in points]

    source_times = np.asarray(timestamps, dtype=np.float64)
    if np.allclose(np.diff(source_times), 0):
        return [dict(point) for point in points]

    source_x = np.asarray([point["cx"] for point in points], dtype=np.float64)
    source_y = np.asarray([point["cy"] for point in points], dtype=np.float64)
    observed = np.asarray([1.0 if point.get("observed") else 0.0 for point in points], dtype=np.float64)

    step = 1.0 / target_fps
    target_times = np.arange(source_times[0], source_times[-1] + (step * 0.5), step, dtype=np.float64)
    resampled_x = np.interp(target_times, source_times, source_x)
    resampled_y = np.interp(target_times, source_times, source_y)
    resampled_observed = np.interp(target_times, source_times, observed)

    resampled_points: List[Dict[str, Any]] = []
    for index, timestamp in enumerate(target_times):
        resampled_points.append(
            {
                "frame_index": index,
                "timestamp": float(timestamp),
                "cx": float(resampled_x[index]),
                "cy": float(resampled_y[index]),
                "observed": bool(resampled_observed[index] >= 0.5),
                "conf": None,
            }
        )
    return resampled_points


def _maybe_resample_segment(segment: Dict[str, Any], target_fps: Optional[float]) -> Dict[str, Any]:
    segment = copy.deepcopy(segment)
    if target_fps is None or target_fps <= 0:
        return segment
    source_fps = float(segment.get("fps") or 0.0)
    if source_fps > 0 and abs(source_fps - target_fps) < 1e-6:
        return segment
    segment["points"] = _resample_with_timestamps(segment.get("points", []), target_fps)
    segment["fps"] = float(target_fps)
    segment["length"] = len(segment["points"])
    return segment


def _load_segments_from_pt(input_path: str | Path) -> List[Dict[str, Any]]:
    payload = torch.load(Path(input_path), map_location="cpu")
    if isinstance(payload, dict) and "segments" in payload:
        return list(payload["segments"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"Unsupported track segment payload: {input_path}")


def _load_segments_from_csv(input_path: str | Path) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_name = row.get("video_name") or row.get("video_stem") or "unknown"
            segment_index = int(row.get("segment_index") or 0)
            key = (video_name, segment_index)
            grouped.setdefault(
                key,
                {
                    "video_name": video_name,
                    "video_stem": Path(video_name).stem,
                    "video_path": row.get("video_path") or "",
                    "segment_index": segment_index,
                    "fps": float(row.get("fps") or 0.0),
                    "points": [],
                },
            )
            grouped[key]["points"].append(
                {
                    "frame_index": int(row.get("frame_index") or len(grouped[key]["points"])),
                    "timestamp": float(row["timestamp"]) if row.get("timestamp") not in (None, "") else None,
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "observed": bool(int(row.get("observed") or 1)),
                    "conf": float(row["conf"]) if row.get("conf") not in (None, "") else None,
                }
            )
    segments = list(grouped.values())
    for segment in segments:
        segment["points"].sort(key=lambda item: item["frame_index"])
        segment["length"] = len(segment["points"])
    return segments


def load_track_segments(input_path: str | Path, target_fps: Optional[float] = None) -> List[Dict[str, Any]]:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".pt":
        segments = _load_segments_from_pt(input_path)
    elif suffix == ".csv":
        segments = _load_segments_from_csv(input_path)
    else:
        raise ValueError(f"Unsupported track segment input: {input_path}")
    return resample_track_segments(segments, target_fps=target_fps)


def resample_track_segments(
    segments: Sequence[Dict[str, Any]],
    target_fps: Optional[float] = None,
) -> List[Dict[str, Any]]:
    return [_maybe_resample_segment(segment, target_fps) for segment in segments]


def build_samples_from_segment(
    segment: Dict[str, Any],
    config: SampleBuildConfig,
    baseline_generator: Optional[AKFBaselineGenerator] = None,
) -> List[Dict[str, Any]]:
    points = segment.get("points", [])
    if len(points) < config.min_history + config.future_steps:
        return []

    xy = _to_float_tensor([[point["cx"], point["cy"]] for point in points])
    baseline_generator = baseline_generator or AKFBaselineGenerator(
        estimator_config=_build_estimator_config(config)
    )

    samples: List[Dict[str, Any]] = []
    max_start = xy.shape[0] - config.future_steps
    for current_index in range(config.min_history, max_start + 1):
        history_start = max(0, current_index - config.max_history)
        history_xy = xy[history_start:current_index]
        history_len = int(history_xy.shape[0])
        if history_len < config.min_history:
            continue
        target_xy = xy[current_index : current_index + config.future_steps]
        baseline_xy = baseline_generator.predict_future(history_xy, config.future_steps)
        sample_index = len(samples)
        samples.append(
            {
                "history_xy": history_xy,
                "history_len": history_len,
                "baseline_xy": baseline_xy,
                "target_xy": target_xy,
                "meta": {
                    "video_name": segment.get("video_name"),
                    "video_stem": segment.get("video_stem"),
                    "video_path": segment.get("video_path") or "",
                    "fps": float(segment.get("fps") or 0.0),
                    "segment_index": int(segment.get("segment_index", 0)),
                    "sample_index": sample_index,
                    "start_frame_index": points[history_start]["frame_index"],
                    "history_end_frame_index": points[current_index - 1]["frame_index"],
                    "target_end_frame_index": points[current_index + config.future_steps - 1]["frame_index"],
                    "history_frame_indices": [
                        int(point.get("frame_index", index))
                        for index, point in enumerate(points[history_start:current_index], start=history_start)
                    ],
                    "target_frame_indices": [
                        int(point.get("frame_index", index))
                        for index, point in enumerate(
                            points[current_index : current_index + config.future_steps],
                            start=current_index,
                        )
                    ],
                    "history_timestamps": [
                        None if point.get("timestamp") is None else float(point["timestamp"])
                        for point in points[history_start:current_index]
                    ],
                    "target_timestamps": [
                        None if point.get("timestamp") is None else float(point["timestamp"])
                        for point in points[current_index : current_index + config.future_steps]
                    ],
                },
            }
        )
    return samples


def build_samples_from_segments(
    segments: Sequence[Dict[str, Any]],
    config: SampleBuildConfig,
    baseline_generator: Optional[AKFBaselineGenerator] = None,
) -> List[Dict[str, Any]]:
    baseline_generator = baseline_generator or AKFBaselineGenerator(
        estimator_config=_build_estimator_config(config)
    )
    samples: List[Dict[str, Any]] = []
    for segment in segments:
        samples.extend(build_samples_from_segment(segment, config, baseline_generator))
    return samples


def iter_json_samples_from_segments(
    segments: Sequence[Dict[str, Any]],
    config: SampleBuildConfig,
) -> Iterable[Tuple[str, Dict[str, List[List[float]]]]]:
    baseline_generator = AKFBaselineGenerator(estimator_config=_build_estimator_config(config))
    for segment in segments:
        samples = build_samples_from_segment(segment, config, baseline_generator)
        video_stem = _sanitize_token(str(segment.get("video_stem") or segment.get("video_name") or "sample"))
        segment_index = int(segment.get("segment_index", 0))
        for sample in samples:
            sample_index = int(sample["meta"]["sample_index"])
            filename = f"{video_stem}_seg{segment_index}_sam{sample_index:06d}.json"
            payload = {
                "input": _format_xy_sequence(sample["history_xy"]),
                "target": _format_xy_sequence(sample["target_xy"]),
                "meta": sample["meta"],
            }
            yield filename, payload


def export_json_samples(
    segments: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    config: SampleBuildConfig,
) -> int:
    records = list(iter_json_samples_from_segments(segments, config))
    return export_json_records(records, output_dir)


def export_json_records(
    records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    output_dir: str | Path,
) -> int:
    output_dir = _prepare_json_output_dir(output_dir)
    sample_count = 0
    for filename, payload in records:
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        sample_count += 1
    return sample_count


def split_json_records(
    records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "sample",
) -> Tuple[List[Tuple[str, Dict[str, List[List[float]]]]], List[Tuple[str, Dict[str, List[List[float]]]]]]:
    records = list(records)
    if not records:
        return [], []
    if val_ratio <= 0:
        return records, []

    split_mode = str(split_mode).strip().lower()
    val_count = int(len(records) * val_ratio)
    if val_ratio > 0 and len(records) > 1:
        val_count = max(1, val_count)
        val_count = min(len(records) - 1, val_count)

    if split_mode == "sample":
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        val_records = shuffled[:val_count]
        train_records = shuffled[val_count:]
        return train_records, val_records

    grouped = _group_json_records(records, split_mode)
    if len(grouped) < 2:
        raise ValueError(
            f"Split mode '{split_mode}' requires at least two groups, got {len(grouped)}."
        )

    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    train_records: List[Tuple[str, Dict[str, List[List[float]]]]] = []
    val_records: List[Tuple[str, Dict[str, List[List[float]]]]] = []
    train_group_keys: List[Tuple[str, ...]] = []
    val_group_keys: List[Tuple[str, ...]] = []
    train_count = 0
    assigned_val_count = 0
    total_count = len(records)
    target_train_count = total_count - val_count
    remaining_count = total_count

    for group_key, group_records in group_items:
        group_size = len(group_records)
        remaining_count -= group_size
        must_go_val = assigned_val_count + remaining_count < val_count
        must_go_train = train_count + remaining_count < target_train_count
        assign_to_val = False
        if must_go_val:
            assign_to_val = True
        elif must_go_train:
            assign_to_val = False
        else:
            gap_if_val = abs((assigned_val_count + group_size) - val_count)
            gap_if_train = abs(assigned_val_count - val_count)
            assign_to_val = gap_if_val <= gap_if_train

        if assign_to_val:
            val_records.extend(group_records)
            val_group_keys.append(group_key)
            assigned_val_count += group_size
        else:
            train_records.extend(group_records)
            train_group_keys.append(group_key)
            train_count += group_size

    if not val_records and train_group_keys:
        fallback_key = train_group_keys.pop()
        fallback_records = grouped[fallback_key]
        train_records = [record for record in train_records if record not in fallback_records]
        val_records.extend(fallback_records)
        val_group_keys.append(fallback_key)
    elif not train_records and val_group_keys:
        fallback_key = val_group_keys.pop()
        fallback_records = grouped[fallback_key]
        val_records = [record for record in val_records if record not in fallback_records]
        train_records.extend(fallback_records)
        train_group_keys.append(fallback_key)

    return train_records, val_records


def build_split_manifest(
    records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    train_records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    val_records: Sequence[Tuple[str, Dict[str, List[List[float]]]]],
    *,
    split_mode: str,
    val_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    all_groups = _group_json_records(records, split_mode)
    train_groups = _group_json_records(train_records, split_mode)
    val_groups = _group_json_records(val_records, split_mode)
    return {
        "split_mode": split_mode,
        "val_ratio": float(val_ratio),
        "split_seed": int(seed),
        "total_records": len(records),
        "total_groups": len(all_groups),
        "train": {
            "record_count": len(train_records),
            "group_count": len(train_groups),
            "groups": [
                {
                    **_serialize_group_key(group_key),
                    "sample_count": len(group_records),
                }
                for group_key, group_records in sorted(train_groups.items())
            ],
        },
        "val": {
            "record_count": len(val_records),
            "group_count": len(val_groups),
            "groups": [
                {
                    **_serialize_group_key(group_key),
                    "sample_count": len(group_records),
                }
                for group_key, group_records in sorted(val_groups.items())
            ],
        },
    }


def export_json_splits(
    segments: Sequence[Dict[str, Any]],
    train_output_dir: str | Path,
    val_output_dir: str | Path | None,
    config: SampleBuildConfig,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    split_mode: str = "segment",
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    records = list(iter_json_samples_from_segments(segments, config))
    train_records, val_records = split_json_records(
        records,
        val_ratio=val_ratio,
        seed=split_seed,
        split_mode=split_mode,
    )

    train_count = export_json_records(train_records, train_output_dir)
    val_count = 0
    if val_output_dir is not None:
        if val_records:
            val_count = export_json_records(val_records, val_output_dir)
        else:
            _prepare_json_output_dir(val_output_dir)

    manifest = build_split_manifest(
        records,
        train_records,
        val_records,
        split_mode=split_mode,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    saved_manifest_path = _write_split_manifest(manifest, manifest_path)

    return {
        "total": len(records),
        "train": train_count,
        "val": val_count,
        "split_mode": split_mode,
        "manifest_path": str(saved_manifest_path) if saved_manifest_path is not None else "",
    }


def save_samples(
    samples: Sequence[Dict[str, Any]],
    output_path: str | Path,
    config: SampleBuildConfig,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "samples": list(samples),
    }
    torch.save(payload, output_path)


class SavedSampleDataset(Dataset):
    def __init__(
        self,
        sample_path: str | Path,
        return_meta: bool = False,
        **config_overrides: Any,
    ) -> None:
        sample_path = Path(sample_path)
        payload = torch.load(sample_path, map_location="cpu")
        if isinstance(payload, dict):
            payload_config = payload.get("config", {})
            self.samples = list(payload.get("samples", []))
        else:
            payload_config = {}
            self.samples = list(payload)
        stored_config = _resolve_sample_config(payload_config)
        self.config = _resolve_sample_config(payload_config, **config_overrides)
        self.return_meta = return_meta
        self.max_history = int(self.config.get("max_history", 15))
        self.future_steps = int(self.config.get("future_steps", 15))
        self.rebuild_baseline = self.config != stored_config
        self.baseline_generator = AKFBaselineGenerator(
            estimator_config=_build_estimator_config(self.config)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        history_xy = sample["history_xy"]
        if not isinstance(history_xy, torch.Tensor):
            history_xy = _to_float_tensor(history_xy)
        history_len = int(sample.get("history_len", history_xy.shape[0]))
        history_xy = history_xy[-history_len:]
        padded_history, history_len = _left_pad_history(history_xy, self.max_history)

        target_xy = sample["target_xy"]
        if not isinstance(target_xy, torch.Tensor):
            target_xy = _to_float_tensor(target_xy)
        target_xy = target_xy[: self.future_steps]

        baseline_xy = sample.get("baseline_xy")
        if baseline_xy is None or self.rebuild_baseline:
            baseline_xy = self.baseline_generator.predict_future(history_xy, self.future_steps)
        elif not isinstance(baseline_xy, torch.Tensor):
            baseline_xy = _to_float_tensor(baseline_xy)

        item: Dict[str, Any] = {
            "history_xy": padded_history,
            "history_len": torch.tensor(history_len, dtype=torch.long),
            "baseline_xy": baseline_xy.to(dtype=torch.float32),
            "target_xy": target_xy.to(dtype=torch.float32),
        }
        if self.return_meta and "meta" in sample:
            item["meta"] = sample["meta"]
        return item


class JsonSampleDataset(Dataset):
    def __init__(
        self,
        sample_path: str | Path,
        min_history: int = 5,
        max_history: int = 15,
        future_steps: int = 15,
        return_meta: bool = False,
        dt: float = 1.0,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        initial_covariance: float = 1.0,
        adaptive_measurement_noise: bool = True,
        measurement_noise_min_scale: float = 0.25,
        measurement_noise_max_scale: float = 16.0,
        innovation_gain: float = 1.0,
        innovation_smoothing: float = 0.2,
        confidence_gain: float = 1.0,
        missing_process_noise_scale: float = 1.35,
        max_process_noise_scale: float = 6.0,
    ) -> None:
        sample_path = Path(sample_path)
        if sample_path.is_dir():
            self.sample_files = sorted(sample_path.glob("*.json"))
        else:
            self.sample_files = [sample_path]
        self.config = _resolve_sample_config(
            min_history=min_history,
            max_history=max_history,
            future_steps=future_steps,
            dt=dt,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=initial_covariance,
            adaptive_measurement_noise=adaptive_measurement_noise,
            measurement_noise_min_scale=measurement_noise_min_scale,
            measurement_noise_max_scale=measurement_noise_max_scale,
            innovation_gain=innovation_gain,
            innovation_smoothing=innovation_smoothing,
            confidence_gain=confidence_gain,
            missing_process_noise_scale=missing_process_noise_scale,
            max_process_noise_scale=max_process_noise_scale,
        )
        self.min_history = int(self.config["min_history"])
        self.max_history = int(self.config["max_history"])
        self.future_steps = int(self.config["future_steps"])
        self.return_meta = return_meta
        self.baseline_generator = AKFBaselineGenerator(
            estimator_config=_build_estimator_config(self.config)
        )

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_path = self.sample_files[index]
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        history_xy = _to_float_tensor(payload["input"])
        target_xy = _to_float_tensor(payload["target"])[: self.future_steps]
        padded_history, history_len = _left_pad_history(history_xy, self.max_history)
        baseline_xy = self.baseline_generator.predict_future(history_xy, self.future_steps)

        item: Dict[str, Any] = {
            "history_xy": padded_history,
            "history_len": torch.tensor(history_len, dtype=torch.long),
            "baseline_xy": baseline_xy,
            "target_xy": target_xy,
        }
        if self.return_meta:
            item["meta"] = {
                "sample_path": str(sample_path),
                **dict(payload.get("meta") or {}),
            }
        return item


def build_and_save_samples(
    input_path: str | Path,
    output_path: str | Path,
    config: SampleBuildConfig,
) -> None:
    segments = load_track_segments(input_path, target_fps=config.target_fps)
    samples = build_samples_from_segments(segments, config)
    save_samples(samples, output_path, config)


def build_and_export_json_samples(
    input_path: str | Path,
    train_output_dir: str | Path,
    val_output_dir: str | Path | None,
    config: SampleBuildConfig,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    split_mode: str = "segment",
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    segments = load_track_segments(input_path, target_fps=config.target_fps)
    return export_json_splits(
        segments,
        train_output_dir=train_output_dir,
        val_output_dir=val_output_dir,
        config=config,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_mode=split_mode,
        manifest_path=manifest_path,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_dataset_from_videos(
    videos_dir: str | Path,
    model_path: str | Path,
    train_output_dir: str | Path,
    val_output_dir: str | Path | None,
    config: SampleBuildConfig,
    confidence_threshold: float = 0.3,
    max_missing: int = 5,
    min_segment_length: int = 5,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    split_mode: str = "segment",
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    detections = collect_detections(videos_dir, model_path, confidence_threshold)
    track_config = TrackBuildConfig(
        max_missing=max_missing,
        min_segment_length=min_segment_length,
        dt=config.dt,
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
    segments = build_track_segments(detections, track_config)
    segments = resample_track_segments(segments, target_fps=config.target_fps)
    return export_json_splits(
        segments,
        train_output_dir=train_output_dir,
        val_output_dir=val_output_dir,
        config=config,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_mode=split_mode,
        manifest_path=manifest_path,
    )


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build JSON samples from videos or prebuilt tracks.")
    parser.add_argument("--videos-dir", type=Path, default=root / "data" / "video")
    parser.add_argument("--model-path", type=Path, default=root / "weights" / "yolo" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Legacy alias for --train-output-dir.")
    parser.add_argument("--train-output-dir", type=Path, default=root / "data" / "dataset" / "train")
    parser.add_argument("--val-output-dir", type=Path, default=root / "data" / "dataset" / "val")
    parser.add_argument("--tracks-input", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--max-missing", type=int, default=5)
    parser.add_argument("--min-segment-length", type=int, default=5)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=15)
    parser.add_argument("--future-steps", type=int, default=15)
    parser.add_argument("--target-fps", type=float, default=40.0)
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
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Random validation split ratio in [0, 1).")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed used for train/val split.")
    parser.add_argument(
        "--split-mode",
        choices=["sample", "segment", "video"],
        default="segment",
        help="How to split train/val records. 'segment' keeps whole track segments together.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional path for saving the resolved train/val split manifest as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_output_dir = args.output_dir or args.train_output_dir
    val_output_dir = None if args.val_ratio <= 0 else args.val_output_dir
    split_manifest = args.split_manifest or (Path(train_output_dir).parent / "split_manifest.json")
    config = SampleBuildConfig(
        min_history=args.min_history,
        max_history=args.max_history,
        future_steps=args.future_steps,
        target_fps=args.target_fps,
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
    if args.tracks_input is not None:
        summary = build_and_export_json_samples(
            args.tracks_input,
            train_output_dir=train_output_dir,
            val_output_dir=val_output_dir,
            config=config,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            split_mode=args.split_mode,
            manifest_path=split_manifest,
        )
        print(
            f"Dataset ready | train={summary['train']} | val={summary['val']} | total={summary['total']}",
            flush=True,
        )
        print(f"Train dir: {train_output_dir}", flush=True)
        if val_output_dir is not None:
            print(f"Val dir: {val_output_dir}", flush=True)
        if summary.get("manifest_path"):
            print(f"Split manifest: {summary['manifest_path']}", flush=True)
        return
    summary = build_dataset_from_videos(
        videos_dir=args.videos_dir,
        model_path=args.model_path,
        train_output_dir=train_output_dir,
        val_output_dir=val_output_dir,
        config=config,
        confidence_threshold=args.confidence_threshold,
        max_missing=args.max_missing,
        min_segment_length=args.min_segment_length,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        manifest_path=split_manifest,
    )
    print(
        f"Dataset ready | train={summary['train']} | val={summary['val']} | total={summary['total']}",
        flush=True,
    )
    print(f"Train dir: {train_output_dir}", flush=True)
    if val_output_dir is not None:
        print(f"Val dir: {val_output_dir}", flush=True)
    if summary.get("manifest_path"):
        print(f"Split manifest: {summary['manifest_path']}", flush=True)


__all__ = [
    "AKFBaselineGenerator",
    "JsonSampleDataset",
    "SampleBuildConfig",
    "SavedSampleDataset",
    "build_and_export_json_samples",
    "build_and_save_samples",
    "build_dataset_from_videos",
    "build_samples_from_segment",
    "build_samples_from_segments",
    "export_json_records",
    "export_json_samples",
    "export_json_splits",
    "iter_json_samples_from_segments",
    "load_track_segments",
    "main",
    "parse_args",
    "resample_track_segments",
    "save_samples",
    "split_json_records",
]
