"""CPU ONNX computer vision. No synthetic detections or silent model fallback."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .tracking import (Detection, EpisodeFilter, IoUTracker, assign_helmet_status,
                       intersection_over_union, normalize_polygon, point_in_polygon)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/safetyvision_yolov8n.onnx"
DEFAULT_PERSON_MODEL = PROJECT_ROOT / "models/yolov8n_coco.onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def non_maximum_suppression(detections: list[Detection], threshold: float = 0.45) -> list[Detection]:
    """Class-aware NMS after confidence filtering."""
    remaining = sorted(detections, key=lambda d: d.confidence, reverse=True)
    selected = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [d for d in remaining if d.label != current.label
                     or intersection_over_union(d.box, current.box) <= threshold]
    return selected


class OnnxDetector:
    def __init__(self, model_path=DEFAULT_MODEL, confidence=0.35, nms_iou=0.45, threads=2):
        import onnxruntime as ort
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model missing: {self.model_path}. Run python scripts/fetch_model.py")
        if not 0 < confidence < 1:
            raise ValueError("confidence must lie between 0 and 1")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if len(inp.shape) != 4 or inp.shape[1] != 3:
            raise ValueError("Expected an RGB NCHW object-detection model")
        self.input_name = inp.name
        self.height, self.width = int(inp.shape[2]), int(inp.shape[3])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "names" not in metadata:
            # The publisher's COCO export has no ONNX metadata. Only accept its
            # separately sourced labels when the model's pinned hash matches.
            manifest_path = PROJECT_ROOT/"models/PERSON_MODEL_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            if _sha256(self.model_path) != manifest["sha256"] or not manifest.get("class_names"):
                raise ValueError("Model needs verified class-name metadata")
            names = manifest["class_names"]
            metadata = dict(metadata, names_source=manifest["class_names_source"])
        else:
            names = ast.literal_eval(metadata["names"])
        self.names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        self.confidence, self.nms_iou = confidence, nms_iou
        self.metadata = metadata

    def predict(self, frame) -> list[Detection]:
        import cv2
        import numpy as np
        original_h, original_w = frame.shape[:2]
        ratio = min(self.width/original_w, self.height/original_h)
        resized_w, resized_h = round(original_w*ratio), round(original_h*ratio)
        left, top = (self.width-resized_w)//2, (self.height-resized_h)//2
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[top:top+resized_h, left:left+resized_w] = resized
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255.0
        prediction = self.session.run(None, {self.input_name: tensor})[0]
        if prediction.ndim != 3 or prediction.shape[0] != 1:
            raise ValueError(f"Unsupported detector output shape: {prediction.shape}")
        rows = prediction[0]
        channels = 4 + len(self.names)
        if rows.shape[0] == channels:
            rows = rows.T
        elif rows.shape[1] != channels:
            raise ValueError(f"Expected YOLOv8/11 xywh+classes output, received {prediction.shape}")
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), class_ids+4]
        detections = []
        for row, class_id, score in zip(rows[scores >= self.confidence], class_ids[scores >= self.confidence], scores[scores >= self.confidence]):
            label = self.names[int(class_id)]
            if label.lower().replace("_", "-") not in {"person", "hardhat", "helmet", "no-hardhat", "no-helmet"}:
                continue
            cx, cy, w, h = map(float, row[:4])
            box = ((cx-w/2-left)/ratio, (cy-h/2-top)/ratio,
                   (cx+w/2-left)/ratio, (cy+h/2-top)/ratio)
            clipped = (max(0.0, min(float(original_w), box[0])), max(0.0, min(float(original_h), box[1])),
                       max(0.0, min(float(original_w), box[2])), max(0.0, min(float(original_h), box[3])))
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                detections.append(Detection(clipped, float(score), label))
        return non_maximum_suppression(detections, self.nms_iou)


def annotate_frame(frame, detections, observations, polygon=None, source_type="video"):
    import cv2
    import numpy as np
    output = frame.copy()
    scale = max(0.45, min(frame.shape[1]/1500, 0.8))
    if polygon:
        cv2.polylines(output, [np.array(polygon, dtype=np.int32)], True, (60, 125, 245), 2)
    colours = {"person": (225, 185, 50), "hardhat": (105, 210, 55), "no-hardhat": (70, 100, 245)}
    for detection in detections:
        x1, y1, x2, y2 = map(round, detection.box)
        colour = colours.get(detection.label.lower(), (180, 180, 180))
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th+5, y1)
        tx = min(x1, max(0, output.shape[1]-tw-6))
        cv2.rectangle(output, (tx, ty-th-5), (tx+tw+5, ty+baseline), (17, 25, 31), -1)
        cv2.putText(output, text, (tx+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    banner = "MODEL PREDICTIONS | PUBLIC IMAGE | not verified incidents" if source_type == "public_image" else "MODEL PREDICTIONS | camera-local tracks"
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (17, 25, 31), -1)
    cv2.putText(output, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, min(scale, 0.52), (230, 235, 240), 1, cv2.LINE_AA)
    return output


def analyze_source(source, output_dir, model_path=DEFAULT_MODEL, camera_id="public-camera",
                   session_id=None, zone_id="unconfigured", polygon=None, sample_fps=2.0,
                   max_frames=120, confidence=0.35, start_time=None, threads=2,
                   person_model_path=DEFAULT_PERSON_MODEL) -> dict:
    """Run actual inference on one image or video and write auditable JSONL.

    Still images receive nominal duration_s=1, duration_is_measured=False and no
    temporal incident IDs. Video timestamps use media offsets from start_time;
    when omitted, start_time is processing time, not a claimed capture date.
    """
    import cv2
    import numpy as np
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if sample_fps <= 0 or max_frames <= 0:
        raise ValueError("sample_fps and max_frames must be positive")
    image_input = source.suffix.lower() in IMAGE_SUFFIXES
    source_type = "public_image" if image_input else "video"
    source_digest = _sha256(source)
    started = datetime.now(timezone.utc)
    session_id = session_id or f"vision-{source_digest[:8]}-{started.strftime('%Y%m%dT%H%M%S%f')}"
    base_time = datetime.fromisoformat(start_time.replace("Z", "+00:00")) if start_time else started
    if base_time.tzinfo is None:
        raise ValueError("start_time must contain a UTC offset")
    base_time = base_time.astimezone(timezone.utc)
    detector = OnnxDetector(model_path, confidence=confidence, threads=threads)
    person_detector = OnnxDetector(person_model_path, confidence=confidence, threads=threads) if person_model_path else None
    tracker = IoUTracker()
    helmet_filter = EpisodeFilter(f"{camera_id}:{session_id}", confirmation_s=1.0)
    zone_filter = EpisodeFilter(f"{camera_id}:{session_id}", confirmation_s=0.0)
    capture = None
    if image_input:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot decode image: {source}")
        frames = [(0, 0.0, 1.0, frame)]
    else:
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot open video with a valid frame rate: {source}")
        stride = max(1, round(fps/sample_fps))
        total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        def video_frames():
            frame_number, sampled = 0, 0
            while sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % stride == 0:
                    represented_frames = min(stride, total_frames-frame_number) if total_frames > 0 else stride
                    yield frame_number, frame_number/fps, max(1, represented_frames)/fps, frame
                    sampled += 1
                frame_number += 1
        frames = video_frames()
    observations, frame_results, assets, latencies = [], [], [], []
    try:
        for frame_number, offset_s, duration_s, frame in frames:
            clock = time.perf_counter()
            detections = detector.predict(frame)
            if person_detector:
                detections = ([d for d in detections if d.label.lower() != "person"]
                              + [d for d in person_detector.predict(frame) if d.label.lower() == "person"])
            latency = (time.perf_counter()-clock)*1000
            latencies.append(latency)
            people = [d for d in detections if d.label.lower() == "person"]
            helmets = [d for d in detections if d.label.lower() != "person"]
            statuses = assign_helmet_status(people, helmets)
            tracks = tracker.update(people, offset_s)
            pixel_polygon = normalize_polygon(polygon, frame.shape[1], frame.shape[0]) if polygon else None
            evidence = output_dir/f"{source.stem}_frame_{frame_number:06d}.jpg"
            frame_observations = []
            for (track_id, person), (status, status_confidence) in zip(tracks, statuses):
                x1, y1, x2, y2 = person.box
                # A truncated person at the image bottom has an uncertain ground point.
                feet_visible = y2 < frame.shape[0]-2
                inside = bool(feet_visible and point_in_polygon(((x1+x2)/2, y2), pixel_polygon))
                helmet_incident = None if image_input else helmet_filter.update(track_id, "helmet", None if status == "unknown" else status == "absent", offset_s)
                zone_incident = None if image_input else zone_filter.update(track_id, "zone", inside if feet_visible else None, offset_s)
                row = {
                    "event_id": f"{camera_id}:{session_id}:{frame_number}:{track_id}",
                    "event_time": (base_time+timedelta(seconds=offset_s)).isoformat().replace("+00:00", "Z"),
                    "camera_id": camera_id, "session_id": session_id, "track_id": track_id,
                    "zone_id": zone_id, "helmet_status": status, "in_restricted_zone": inside,
                    "confidence": round(min(person.confidence, status_confidence) if status != "unknown" else person.confidence, 6),
                    "duration_s": duration_s, "duration_is_measured": not image_input,
                    "helmet_incident_id": helmet_incident, "zone_incident_id": zone_incident,
                    "evidence_path": _relative(evidence), "source_type": source_type,
                    "event_time_is_assumed": start_time is None, "zone_assessable": bool(pixel_polygon and feet_visible),
                    "person_box": [round(v, 2) for v in person.box], "media_offset_s": offset_s,
                }
                frame_observations.append(row)
            observations.extend(frame_observations)
            annotated = annotate_frame(frame, detections, frame_observations, pixel_polygon, source_type=source_type)
            if not cv2.imwrite(str(evidence), annotated):
                raise OSError(f"Failed to write evidence: {evidence}")
            frame_results.append({"frame_number": frame_number, "media_offset_s": offset_s,
                                  "width": frame.shape[1], "height": frame.shape[0],
                                  "inference_ms": round(latency, 3), "detections": [d.as_dict() for d in detections],
                                  "observations": frame_observations, "evidence_path": _relative(evidence)})
            assets.append({"path": _relative(evidence), "title": source.stem.replace("_", " "),
                           "kind": "image", "source_type": source_type,
                           "description": "Actual CPU model output. Labels are predictions; no accuracy or incident verification is implied."})
    finally:
        if capture:
            capture.release()
    observation_path = output_dir/"observations.jsonl"
    with observation_path.open("w", encoding="utf-8") as f:
        for row in observations:
            f.write(json.dumps(row)+"\n")
    report = {
        "engine": "ONNX Runtime", "execution_provider": "CPUExecutionProvider",
        "model_path": _relative(Path(model_path)), "model_sha256": _sha256(Path(model_path)),
        "person_model_path": _relative(Path(person_model_path)) if person_model_path else None,
        "person_model_sha256": _sha256(Path(person_model_path)) if person_model_path else None,
        "model_metadata": detector.metadata, "confidence_threshold": confidence,
        "source": _relative(source), "source_sha256": source_digest, "source_type": source_type,
        "created_at": started.isoformat(), "frames_processed": len(frame_results),
        "person_observations": len(observations),
        "helmet_status_counts": dict(Counter(r["helmet_status"] for r in observations)),
        "detection_counts": dict(Counter(d["label"] for f in frame_results for d in f["detections"])),
        "inference_latency_ms": {"mean": float(np.mean(latencies)) if latencies else None,
                                 "p95": float(np.percentile(latencies, 95)) if latencies else None},
        "accuracy": {"measured": False, "precision": None, "recall": None, "mAP": None,
                     "reason": "These public photographs have no verified object-level ground-truth annotations."},
        "observation_path": _relative(observation_path), "assets": assets, "frames": frame_results,
        "limitations": ["Pretrained industrial PPE model; not trained or validated on a mine dataset in this project.",
                        "No-helmet needs an explicit model detection; missing or conflicting evidence is unknown.",
                        "Still-image samples have nominal duration=1; they do not measure exposure time or temporal incidents.",
                        "Track IDs are camera-local and can change after occlusion; no real identity is inferred.",
                        "A polygon encodes a configured image region, not calibrated physical distance."]}
    (output_dir/"vision_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"outputs/vision")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--person-model", type=Path, default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--camera-id", default="public-camera")
    parser.add_argument("--zone-id", default="unconfigured")
    parser.add_argument("--session-id")
    parser.add_argument("--polygon", help='Normalized JSON polygon, e.g. [[0.5,0.5],[1,0.5],[1,1],[0.5,1]]')
    parser.add_argument("--sample-fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--start-time", help="Known capture start timestamp with UTC offset; otherwise processing time is assumed")
    args = parser.parse_args()
    report = analyze_source(args.source, args.output, args.model, camera_id=args.camera_id,
                            session_id=args.session_id, zone_id=args.zone_id,
                            polygon=json.loads(args.polygon) if args.polygon else None,
                            sample_fps=args.sample_fps, max_frames=args.max_frames,
                            confidence=args.confidence, start_time=args.start_time,
                            person_model_path=args.person_model)
    print(json.dumps({k: v for k, v in report.items() if k not in {"frames", "model_metadata"}}, indent=2))


if __name__ == "__main__":
    main()
