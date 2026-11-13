"""Descriptive fixed-sample precision/recall, never presented as mine validation."""
from collections import Counter
import argparse
import json
from pathlib import Path

from .tracking import Detection, intersection_over_union
from .vision import DEFAULT_MODEL, PROJECT_ROOT, OnnxDetector, _sha256


def match_detections(predictions, ground_truth, iou_threshold=0.5):
    """Confidence-ordered, one-to-one, same-class IoU matching."""
    labels = {d.label for d in predictions} | {d.label for d in ground_truth}
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in sorted(labels)}
    matched = set()
    for prediction in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        candidates = [(intersection_over_union(prediction.box, gt.box), i)
                      for i, gt in enumerate(ground_truth)
                      if i not in matched and gt.label == prediction.label]
        overlap, index = max(candidates, default=(0, None))
        if index is not None and overlap >= iou_threshold:
            matched.add(index)
            counts[prediction.label]["tp"] += 1
        else:
            counts[prediction.label]["fp"] += 1
    for i, truth in enumerate(ground_truth):
        if i not in matched:
            counts[truth.label]["fn"] += 1
    return counts


def _rates(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return dict(counts, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def evaluate_labelled_sample(annotations, output, model_path=DEFAULT_MODEL,
                            confidence=0.35, iou_threshold=0.5):
    import cv2
    annotations, output = Path(annotations), Path(output)
    metadata = json.loads(annotations.read_text())
    detector = OnnxDetector(model_path, confidence=confidence)
    totals = {name: Counter(tp=0, fp=0, fn=0) for name in metadata["class_names"]}
    per_image = []
    for record in metadata["records"]:
        path = annotations.parent/record["file"]
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Sample hash mismatch: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read sample image: {path}")
        if frame.shape[1] != record["width"] or frame.shape[0] != record["height"]:
            raise ValueError(f"Image/annotation size mismatch: {path}")
        predictions = [Detection(d.box, d.confidence, d.label.lower()) for d in detector.predict(frame)
                       if d.label.lower() in {"hardhat", "no-hardhat"}]
        truth = []
        for box, category in zip(record["boxes_xywh"], record["categories"]):
            x, y, w, h = box
            truth.append(Detection((x, y, x+w, y+h), 1.0, metadata["class_names"][category]))
        counts = match_detections(predictions, truth, iou_threshold)
        for label, values in counts.items():
            totals[label].update(values)
        per_image.append({"file": record["file"], "row_index": record["row_index"],
                          "counts": counts, "predictions": [p.as_dict() for p in predictions],
                          "ground_truth": [g.as_dict() for g in truth]})
    micro = Counter(tp=0, fp=0, fn=0)
    for values in totals.values():
        micro.update(values)
    report = {
        "evaluation_type": "fixed_public_sample_smoke_check",
        "dataset": metadata["dataset"], "dataset_source": metadata["source"],
        "selection": metadata.get("selection"), "images_evaluated": len(per_image),
        "model_sha256": _sha256(Path(model_path)), "confidence_threshold": confidence,
        "iou_threshold": iou_threshold, "per_class": {k: _rates(v) for k, v in totals.items()},
        "micro": _rates(micro), "mAP": None,
        "scope": "Raw helmet/bare-head object localization only; not person association, incident detection, or mine-domain accuracy.",
        "limitations": ["Only the first 20 test rows were chosen before inference; this is a small convenience sample.",
                        "Overlap with the pretrained model's upstream training images is unknown; it is not a verified independent holdout.",
                        "Source images are construction/web data, not a mine-specific evaluation set.",
                        "Precision and recall are computed at the declared threshold; no mAP is claimed."],
        "per_image": per_image,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT/"data/public/hardhat_test_sample/annotations.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"outputs/vision/evaluation.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    report = evaluate_labelled_sample(args.annotations, args.output, args.model, args.confidence)
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
