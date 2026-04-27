from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import guard
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for video export") from _CV2_IMPORT_ERROR


def _require_yolo() -> None:
    if YOLO is None:
        raise ImportError("ultralytics is required for YOLO inference export") from _YOLO_IMPORT_ERROR


def iter_video_files(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def select_best_box(result, confidence_threshold: float):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best_box = None
    best_conf = float("-inf")
    for index in range(len(boxes)):
        conf = float(boxes.conf[index].item())
        if conf < confidence_threshold or conf <= best_conf:
            continue
        best_box = boxes[index]
        best_conf = conf
    return best_box


def _build_detection_row(
    video_path: Path,
    frame_index: int,
    timestamp: float,
    fps: float,
    frame_width: int,
    frame_height: int,
    best_box,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "video_path": str(video_path),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "has_detection": 0,
        "observed": False,
        "conf": "",
        "cx": "",
        "cy": "",
        "w": "",
        "h": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
    }
    if best_box is None:
        return row

    x1, y1, x2, y2 = [float(value) for value in best_box.xyxy[0].tolist()]
    cx = ((x1 + x2) * 0.5) / max(frame_width, 1)
    cy = ((y1 + y2) * 0.5) / max(frame_height, 1)
    width = (x2 - x1) / max(frame_width, 1)
    height = (y2 - y1) / max(frame_height, 1)
    row.update(
        {
            "has_detection": 1,
            "observed": True,
            "conf": float(best_box.conf.item()),
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    )
    return row


def collect_video_detections(
    model,
    video_path: Path,
    confidence_threshold: float,
) -> Tuple[List[Dict[str, object]], int, int]:
    _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    rows: List[Dict[str, object]] = []
    total_frames = 0
    detected_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False, conf=confidence_threshold)
        result = results[0] if results else None
        best_box = select_best_box(result, confidence_threshold) if result is not None else None
        timestamp = total_frames / fps
        row = _build_detection_row(
            video_path=video_path,
            frame_index=total_frames,
            timestamp=timestamp,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            best_box=best_box,
        )
        rows.append(row)
        total_frames += 1
        if best_box is not None:
            detected_frames += 1

    capture.release()
    return rows, total_frames, detected_frames


def export_video_detections(
    model,
    video_path: Path,
    writer: csv.DictWriter,
    confidence_threshold: float,
) -> Tuple[int, int]:
    rows, total_frames, detected_frames = collect_video_detections(
        model=model,
        video_path=video_path,
        confidence_threshold=confidence_threshold,
    )
    for row in rows:
        writer.writerow(row)
    return total_frames, detected_frames


def collect_detections(
    videos_dir: str | Path,
    model_path: str | Path,
    confidence_threshold: float = 0.3,
) -> List[Dict[str, object]]:
    _require_yolo()
    video_files = iter_video_files(videos_dir)
    model = YOLO(str(model_path))

    rows: List[Dict[str, object]] = []
    for video_path in video_files:
        video_rows, _, _ = collect_video_detections(model, video_path, confidence_threshold)
        rows.extend(video_rows)
    return rows


def export_detections(
    videos_dir: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    confidence_threshold: float = 0.3,
) -> None:
    _require_yolo()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    video_files = iter_video_files(videos_dir)
    fieldnames = [
        "video_name",
        "video_stem",
        "video_path",
        "frame_index",
        "timestamp",
        "fps",
        "frame_width",
        "frame_height",
        "has_detection",
        "conf",
        "cx",
        "cy",
        "w",
        "h",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video_path in video_files:
            export_video_detections(
                model=model,
                video_path=video_path,
                writer=writer,
                confidence_threshold=confidence_threshold,
            )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Export YOLO detections from videos.")
    parser.add_argument("--videos-dir", type=Path, default=root / "data" / "video")
    parser.add_argument("--model-path", type=Path, default=root / "weights" / "yolo" / "best.pt")
    parser.add_argument("--output-csv", type=Path, default=root / "data" / "detections.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_detections(
        videos_dir=args.videos_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        confidence_threshold=args.confidence_threshold,
    )


__all__ = [
    "VIDEO_EXTENSIONS",
    "collect_detections",
    "collect_video_detections",
    "export_detections",
    "export_video_detections",
    "iter_video_files",
    "main",
    "parse_args",
    "select_best_box",
]
