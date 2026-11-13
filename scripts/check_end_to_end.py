"""Exercise real API/Spark streaming and save runtime results to outputs/end_to_end.json."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from mine_safety.server import app

client = TestClient(app)
start = time.perf_counter()
result = {"kind": "API and real Spark streaming integration", "source_type": "synthetic", "checks": {}}
try:
    assert client.get("/").status_code == 200
    assert client.get("/web/app.js").status_code == 200
    initial = client.get("/api/status").json()
    result["checks"]["html_and_assets"] = "passed"
    if initial["assets"]:
        assert client.get(initial["assets"][0]["url"]).status_code == 200
        result["checks"]["real_vision_asset_route"] = "passed"
    response = client.post("/api/demo/start")
    assert response.status_code == 200, response.text
    result["start_response"] = response.json()
    assert client.post("/api/demo/start").status_code == 409
    result["checks"]["concurrent_start_rejected"] = "passed"
    snapshots = []
    last_count = -1
    while time.perf_counter() - start < 150:
        data = client.get("/api/status").json()
        if data.get("error"):
            raise AssertionError(data["error"])
        count = data["summary"].get("observations", 0)
        if count != last_count:
            snapshots.append({"elapsed_s": round(time.perf_counter() - start, 3), "observations": count})
            last_count = count
            print(json.dumps(snapshots[-1]), flush=True)
        if count == 720:
            assert data["summary"]["helmet_incidents"] == 5
            assert data["summary"]["zone_incidents"] == 4
            result["summary"] = data["summary"]
            result["metrics"] = data["metrics"]
            result["checks"]["streaming_truth_totals"] = "passed"
            break
        time.sleep(1)
    else:
        raise AssertionError("Streaming did not consume all 720 records within 150 seconds")
    result["snapshots"] = snapshots
    response = client.post("/api/demo/stop")
    assert response.status_code == 200
    for _ in range(15):
        if not client.get("/api/status").json()["running"]:
            break
        time.sleep(1)
    assert not client.get("/api/status").json()["running"]
    result["checks"]["stop_releases_job"] = "passed"
    result["passed"] = True
finally:
    client.post("/api/demo/stop")
    result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    path = ROOT / "outputs/end_to_end.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result.get("passed", False), "report": str(path), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
