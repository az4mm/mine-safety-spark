"""Small deterministic camera-local tracker and conservative safety state rules.

This is an IoU baseline, not cross-camera re-identification. A missed head
detection is unknown. None of these rules identify a real person's identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    label: str

    def as_dict(self) -> dict:
        return {"box": [round(v, 2) for v in self.box],
                "confidence": round(self.confidence, 6), "label": self.label}


def intersection_over_union(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def point_in_polygon(point, polygon) -> bool:
    """Ray casting with boundary inclusion; coordinates must share units."""
    if polygon is None or len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i+1) % len(polygon)]
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-8 and min(x1, x2)-1e-8 <= x <= max(x1, x2)+1e-8 and min(y1, y2)-1e-8 <= y <= max(y1, y2)+1e-8:
            return True
        if (y1 > y) != (y2 > y):
            at_x = x1 + (y-y1) * (x2-x1) / (y2-y1)
            if x < at_x:
                inside = not inside
    return inside


def normalize_polygon(polygon: Iterable, width: int, height: int) -> list:
    """Convert a configured [0,1] polygon to pixel coordinates."""
    points = [tuple(map(float, p)) for p in polygon]
    if len(points) < 3 or any(len(p) != 2 for p in points):
        raise ValueError("A restricted-zone polygon needs at least three x,y points")
    if any(not isfinite(v) or v < 0 or v > 1 for p in points for v in p):
        raise ValueError("Zone coordinates must be finite normalized values in [0, 1]")
    return [(x*width, y*height) for x, y in points]


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_gap_s: float = 1.5):
        self.threshold, self.max_gap_s = iou_threshold, max_gap_s
        self.tracks: dict[str, tuple[Detection, float]] = {}
        self.next_id = 1

    def update(self, detections: list[Detection], timestamp_s: float) -> list[tuple[str, Detection]]:
        self.tracks = {k: v for k, v in self.tracks.items()
                       if 0 <= timestamp_s-v[1] <= self.max_gap_s}
        candidates = sorted(
            ((intersection_over_union(old.box, new.box), key, i)
             for key, (old, _) in self.tracks.items()
             for i, new in enumerate(detections)), reverse=True)
        used_tracks, used_detections, matches = set(), set(), {}
        for overlap, key, i in candidates:
            if overlap >= self.threshold and key not in used_tracks and i not in used_detections:
                matches[i] = key
                used_tracks.add(key)
                used_detections.add(i)
        result = []
        for i, detection in enumerate(detections):
            key = matches.get(i)
            if key is None:
                key = f"track-{self.next_id:05d}"
                self.next_id += 1
            self.tracks[key] = (detection, timestamp_s)
            result.append((key, detection))
        return result


def assign_helmet_status(persons: list[Detection], ppe: list[Detection]) -> list[tuple[str, float]]:
    """Assign each head-region object to at most one person.

    Positive absence requires the model's explicit no-hardhat class. Conflicting
    head evidence is unknown; absence of a detection is also unknown.
    """
    assigned: list[list[Detection]] = [[] for _ in persons]
    for head in ppe:
        label = head.label.lower().replace("_", "-")
        if label not in {"hardhat", "helmet", "no-hardhat", "no-helmet"}:
            continue
        cx, cy = (head.box[0]+head.box[2])/2, (head.box[1]+head.box[3])/2
        candidates = []
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person.box
            w, h = max(x2-x1, 1.0), max(y2-y1, 1.0)
            if x1-.05*w <= cx <= x2+.05*w and y1-.05*h <= cy <= y1+.45*h:
                distance = abs(cx-(x1+x2)/2)/w + abs(cy-(y1+.10*h))/h
                candidates.append((distance, i))
        if candidates:
            assigned[min(candidates)[1]].append(head)
    result = []
    for items in assigned:
        positive = [d.confidence for d in items if d.label.lower() in {"hardhat", "helmet"}]
        negative = [d.confidence for d in items if d.label.lower().replace("_", "-") in {"no-hardhat", "no-helmet"}]
        if positive and not negative:
            result.append(("present", max(positive)))
        elif negative and not positive:
            result.append(("absent", max(negative)))
        else:
            result.append(("unknown", 0.0))
    return result


class EpisodeFilter:
    """Consecutive evidence confirms an episode; confirmed IDs survive a short gap."""
    def __init__(self, namespace: str, confirmation_s: float = 1.0, max_gap_s: float = 1.5):
        self.namespace, self.confirmation_s, self.max_gap_s = namespace, confirmation_s, max_gap_s
        self.states: dict[tuple, dict] = {}
        self.counts: dict[tuple, int] = {}

    def update(self, track_id: str, kind: str, active: bool | None, timestamp_s: float) -> str | None:
        key = (track_id, kind)
        state = self.states.get(key)
        if state and timestamp_s-state["last_seen"] > self.max_gap_s:
            self.states.pop(key, None)
            state = None
        if active is False:
            self.states.pop(key, None)
            return None
        if active is None:
            if state:
                state["started"] = None
            return None
        if state is None:
            state = {"started": timestamp_s, "last_seen": timestamp_s, "id": None}
            self.states[key] = state
        if state["started"] is None:
            state["started"] = timestamp_s
        state["last_seen"] = timestamp_s
        if state["id"] is None and timestamp_s-state["started"] >= self.confirmation_s:
            self.counts[key] = self.counts.get(key, 0) + 1
            state["id"] = f"{self.namespace}:{track_id}:{kind}:{self.counts[key]}"
        return state["id"]
