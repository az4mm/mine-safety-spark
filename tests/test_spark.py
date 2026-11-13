"""Integration checks run against Apache Spark, never a pandas substitute."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pyspark")

from mine_safety.spark_jobs import create_spark, run_batch, run_stream, run_window_stream


def observation(event_id, second=0, **changes):
    record = {
        "event_id": event_id, "event_time": f"2026-01-01T00:00:{second:02d}Z",
        "camera_id": "CAM-1", "session_id": "session-a", "track_id": "worker-1",
        "zone_id": "ZONE-1", "helmet_status": "present", "in_restricted_zone": False,
        "confidence": 0.9, "duration_s": 1.0, "helmet_incident_id": None,
        "zone_incident_id": None, "evidence_path": None, "source_type": "synthetic",
    }
    record.update(changes)
    return record


def write_events(path, events):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in events))


@pytest.fixture(scope="module")
def spark():
    session = create_spark("local[2]", "MineSafetyIntegrationTests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory, spark):
    root = tmp_path_factory.mktemp("spark_batch")
    missing = observation("b", 10, helmet_status="absent", duration_s=2.0,
                          helmet_incident_id="CAM-1/session-a/worker-1/helmet/1")
    events = [
        observation("a", 0, duration_s=10.0), missing,
        observation("c", 12, helmet_status="absent", duration_s=3.0,
                    helmet_incident_id="CAM-1/session-a/worker-1/helmet/1"),
        observation("d", 15, helmet_status="unknown", duration_s=5.0),
        observation("e", 20, duration_s=10.0, in_restricted_zone=True,
                    zone_incident_id="CAM-1/session-a/worker-1/zone/1"),
        dict(missing), observation("bad", event_time="not-a-timestamp"),
        observation("image", camera_id="IMAGE-1", source_type="public_image",
                    helmet_status="absent", duration_is_measured=False),
        observation("bad_status", helmet_status="possibly"),
        observation("bad_missing_id", helmet_status="absent"),
    ]
    write_events(root / "events.jsonl", events)
    result = run_batch(root / "events.jsonl", root / "out")
    return root / "out", result


def test_validation_and_global_deduplication(analyzed):
    output, summary = analyzed
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["engine"] == "Apache Spark"
    assert metrics["input_rows"] == 10
    assert metrics["invalid_rows"] == 3
    assert metrics["duplicate_rows"] == 1
    assert summary["observations"] == 6


def test_exposure_weighting_unknown_and_still_images(analyzed):
    _, summary = analyzed
    assert summary["helmet_seconds"] == 20.0
    assert summary["missing_helmet_seconds"] == 5.0
    assert summary["unknown_seconds"] == 5.0
    assert summary["assessable_seconds"] == 25.0
    assert summary["compliance_pct"] == 80.0
    assert summary["unknown_pct"] == 16.67
    assert summary["public_image_observations"] == 1
    assert summary["helmet_absent_observations"] == 3


def test_episode_counts_rules_and_duration(analyzed):
    output, summary = analyzed
    assert summary["helmet_incidents"] == 1
    assert summary["zone_incidents"] == 1
    incidents = json.loads((output / "incidents.json").read_text())
    helmet = next(row for row in incidents if row["type"] == "helmet_missing")
    assert helmet["duration_s"] == 5.0
    assert helmet["start_time"] == "2026-01-01T00:00:10.000Z"
    assert helmet["end_time"] == "2026-01-01T00:00:15.000Z"
    assert helmet["severity"] == "medium"
    image_camera = next(row for row in json.loads((output / "cameras.json").read_text())
                        if row["camera_id"] == "IMAGE-1")
    assert image_camera["compliance_pct"] is None


def test_stream_restart_is_idempotent_across_microbatches(tmp_path, spark):
    incoming, output, checkpoint = (tmp_path / p for p in ("in", "out", "checkpoint"))
    a = observation("a")
    b = observation("b", 1, helmet_status="absent", helmet_incident_id="episode-b")
    write_events(incoming / "batch01.jsonl", [a, b])
    run_stream(incoming, output, checkpoint, available_now=True)
    first = json.loads((output / "summary.json").read_text())
    write_events(incoming / "batch02.jsonl", [dict(b), observation("c", 2),
                                             observation("bad", event_time="bad")])
    run_stream(incoming, output, checkpoint, available_now=True)
    second = json.loads((output / "summary.json").read_text())
    run_stream(incoming, output, checkpoint, available_now=True)
    third = json.loads((output / "summary.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert first["observations"] == 2
    assert second["observations"] == third["observations"] == 3
    assert second["helmet_incidents"] == third["helmet_incidents"] == 1
    assert metrics["input_rows"] == 5
    assert metrics["duplicate_rows"] == 1
    assert metrics["invalid_rows"] == 1
    assert spark.read.parquet(str(output / "events.parquet")).count() == 3


def test_native_watermark_windows_and_late_event(tmp_path, spark):
    incoming, output, checkpoint = (tmp_path / p for p in ("in", "windows", "checkpoint"))
    old = observation("old")
    advance = observation("advance", event_time="2026-01-01T00:02:00Z")
    write_events(incoming / "batch01.jsonl", [old, dict(old), advance])
    run_window_stream(incoming, output, checkpoint, available_now=True)
    rows = spark.read.parquet(str(output)).collect()
    assert sum(row["observations"] for row in rows) == 1
    write_events(incoming / "batch02.jsonl", [
        observation("late", 20),
        observation("advance-again", event_time="2026-01-01T00:03:00Z"),
    ])
    run_window_stream(incoming, output, checkpoint, available_now=True)
    rows = spark.read.parquet(str(output)).collect()
    first_window = next(row for row in rows if row["window"].start.minute == 0)
    assert first_window["observations"] == 1

