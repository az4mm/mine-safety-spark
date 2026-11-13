"""Deterministic, explicitly synthetic observations with independent expected totals."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

BASE_TIME = datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)
ZONES = {"CAM-01": "GALLERY", "CAM-02": "EQUIPMENT", "CAM-03": "SERVICE"}
SCENARIOS = {
    ("CAM-01", "P01"): {"absent": [(20, 40)], "unknown": [], "zone": []},
    ("CAM-01", "P02"): {"absent": [], "unknown": [(40, 65)], "zone": []},
    ("CAM-02", "P01"): {"absent": [(75, 90)], "unknown": [], "zone": [(15, 35), (70, 95)]},
    ("CAM-02", "P02"): {"absent": [(0, 12), (100, 115)], "unknown": [], "zone": []},
    ("CAM-03", "P01"): {"absent": [], "unknown": [(0, 20)], "zone": [(60, 80)]},
    ("CAM-03", "P02"): {"absent": [(45, 65)], "unknown": [(65, 75)], "zone": [(48, 58)]},
}


def _active(intervals: list[tuple[int, int]], second: int) -> tuple[int, int] | None:
    return next(((start, end) for start, end in intervals if start <= second < end), None)


def generate_observations(seconds: int = 120, session_id: str = "demo-001", start_time: datetime | None = None) -> list[dict]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    start_time = start_time or BASE_TIME
    if start_time.tzinfo is None:
        raise ValueError("start_time must include timezone")
    records = []
    for elapsed in range(seconds):
        second, cycle = elapsed % 120, elapsed // 120
        stamp = (start_time + timedelta(seconds=elapsed)).isoformat().replace("+00:00", "Z")
        for (camera, track), scenarios in SCENARIOS.items():
            absent = _active(scenarios["absent"], second)
            unknown = _active(scenarios["unknown"], second)
            zone = _active(scenarios["zone"], second)
            status = "absent" if absent else "unknown" if unknown else "present"
            prefix = f"{session_id}:{camera}:{track}:{cycle}"
            records.append({
                "event_id": f"{session_id}:{camera}:{track}:{elapsed:06d}",
                "event_time": stamp,
                "camera_id": camera,
                "session_id": session_id,
                "track_id": track,
                "zone_id": ZONES[camera],
                "helmet_status": status,
                "in_restricted_zone": zone is not None,
                "confidence": 0.92 if status != "unknown" else 0.25,
                "duration_s": 1.0,
                "duration_is_measured": True,
                "helmet_incident_id": f"{prefix}:helmet:{absent[0]}" if absent else None,
                "zone_incident_id": f"{prefix}:zone:{zone[0]}" if zone else None,
                "evidence_path": None,
                "source_type": "synthetic",
            })
    return records


def expected_summary(records: Iterable[dict]) -> dict:
    """Independent reference calculation used to verify Spark, never serve as Spark fallback."""
    unique = {row["event_id"]: row for row in records}
    rows = list(unique.values())
    total = sum(row["duration_s"] for row in rows)
    present = sum(row["duration_s"] for row in rows if row["helmet_status"] == "present")
    absent = sum(row["duration_s"] for row in rows if row["helmet_status"] == "absent")
    unknown = sum(row["duration_s"] for row in rows if row["helmet_status"] == "unknown")
    assessed = present + absent
    return {
        "observations": len(rows),
        "cameras": len({row["camera_id"] for row in rows}),
        "workers": len({(row["camera_id"], row["session_id"], row["track_id"]) for row in rows}),
        "helmet_incidents": len({row["helmet_incident_id"] for row in rows if row["helmet_incident_id"]}),
        "zone_incidents": len({row["zone_incident_id"] for row in rows if row["zone_incident_id"]}),
        "assessable_seconds": assessed,
        "helmet_seconds": present,
        "missing_helmet_seconds": absent,
        "unknown_seconds": unknown,
        "restricted_seconds": sum(row["duration_s"] for row in rows if row["in_restricted_zone"]),
        "compliance_pct": 100 * present / assessed if assessed else None,
        "unknown_pct": 100 * unknown / total if total else None,
        "source_type": "synthetic",
        "note": "Reference totals from deterministic scenario definitions. These are not measured CV accuracy.",
    }


def atomic_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    temp.replace(path)
    return count


def prepare_demo(output_dir: str | Path = "data/demo", seconds: int = 120) -> dict:
    root = Path(output_dir)
    records = generate_observations(seconds)
    write_jsonl(root / "events.jsonl", records)
    summary = expected_summary(records)
    atomic_json(root / "expected.json", summary)
    atomic_json(root / "scenarios.json", {
        "source_type": "synthetic",
        "description": "Deterministic camera observations, not video model predictions. All intervals are seconds [start,end).",
        "seconds": seconds,
        "scenarios": [{"camera_id": camera, "track_id": track, **rules} for (camera, track), rules in SCENARIOS.items()],
    })
    return summary


def replay(input_dir: str | Path, seconds: int = 120, batch_seconds: int = 5, delay: float = 2.0, session_id: str | None = None, stop_event=None) -> dict:
    if batch_seconds < 1 or delay < 0:
        raise ValueError("batch_seconds must be positive and delay non-negative")
    session_id = session_id or datetime.now(timezone.utc).strftime("replay-%Y%m%dT%H%M%S%f")
    # Playback timestamps use the replay clock; source media time is simulated.
    records = generate_observations(seconds, session_id, datetime.now(timezone.utc))
    chunk_size = batch_seconds * len(SCENARIOS)
    root = Path(input_dir)
    root.mkdir(parents=True, exist_ok=True)
    sent = []
    for index in range(0, len(records), chunk_size):
        if stop_event is not None and stop_event.is_set():
            break
        chunk = records[index:index + chunk_size]
        write_jsonl(root / f"{session_id}-{index:06d}.jsonl", chunk)
        sent.extend(chunk)
        if stop_event is not None:
            if stop_event.wait(delay):
                break
        else:
            time.sleep(delay)
    return expected_summary(sent)
