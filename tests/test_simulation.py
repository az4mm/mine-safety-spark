from datetime import datetime
import pytest
from mine_safety.simulation import generate_observations, expected_summary, prepare_demo


def test_known_scenario_totals():
    summary = expected_summary(generate_observations())
    assert summary["observations"] == 720
    assert summary["workers"] == 6
    assert summary["helmet_incidents"] == 5
    assert summary["zone_incidents"] == 4
    assert summary["missing_helmet_seconds"] == 82
    assert summary["restricted_seconds"] == 75
    assert summary["unknown_seconds"] == 55
    assert summary["helmet_seconds"] == 583
    assert summary["compliance_pct"] == pytest.approx(583 / 665 * 100)


def test_duplicate_delivery_does_not_change_reference():
    records = generate_observations()
    assert expected_summary(records + records[:30]) == expected_summary(records)


def test_unknown_is_not_noncompliance():
    unknown = [r for r in generate_observations() if r["helmet_status"] == "unknown"]
    result = expected_summary(unknown)
    assert result["compliance_pct"] is None
    assert result["unknown_pct"] == 100
    assert result["helmet_incidents"] == 0


def test_temporary_ids_are_namespaced():
    rows = generate_observations()
    assert len({r["event_id"] for r in rows}) == len(rows)
    assert expected_summary(rows)["workers"] == 6
    with pytest.raises(ValueError):
        generate_observations(start_time=datetime(2026, 1, 1))


def test_fixture_is_reproducible(tmp_path):
    prepare_demo(tmp_path)
    first = (tmp_path / "events.jsonl").read_bytes()
    prepare_demo(tmp_path)
    assert first == (tmp_path / "events.jsonl").read_bytes()
