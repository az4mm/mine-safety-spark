"""Local dashboard API. Spark runs in a child process, outside the web server."""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .simulation import prepare_demo, replay

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("MINE_SAFETY_OUTPUT", ROOT / "outputs/current")).resolve()
app = FastAPI(title="Mine Safety Observatory", version="1.0.0", docs_url="/api/docs")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")
LOCK = threading.RLock()
JOB = {"running": False, "mode": "saved results", "kind": None, "error": None, "message": "Ready", "started_at": None}
PROCESS: subprocess.Popen | None = None
STOP_REPLAY = threading.Event()
ACTIVE_OUTPUT = OUTPUT


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _public_asset_url(path: str | Path) -> str | None:
    try:
        file = Path(path)
        if not file.is_absolute():
            file = ROOT / file
        relative = file.resolve().relative_to(ROOT)
        if file.is_file() and relative.parts[0] in {"data", "outputs", "reports"} and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".svg"}:
            return "/api/asset/" + relative.as_posix()
    except (OSError, ValueError):
        pass
    return None


def _assets() -> list[dict]:
    result = []
    manifest = read_json(ROOT / "outputs/vision/manifest.json", {})
    if isinstance(manifest, dict):
        entries = manifest.get("assets", manifest.get("results", []))
    elif isinstance(manifest, list):
        entries = manifest
    else:
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("annotated_path") or entry.get("image_path") or entry.get("path") or entry.get("url")
        url = _public_asset_url(path) if path else None
        if url:
            result.append({**entry, "url": url, "image": url, "title": entry.get("title", Path(path).stem.replace("_", " ")), "kind": "image", "source_type": "public_image"})
    known = {r["url"] for r in result}
    for path in sorted((ROOT / "outputs/vision").glob("**/*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            url = _public_asset_url(path)
            if url and url not in known:
                result.append({"url": url, "image": url, "title": path.stem.replace("_", " "), "kind": "image", "source_type": "public_image", "description": "Actual CPU model output on a public still image. See source and model notes."})
                known.add(url)
    return result[:30]


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "mine-safety-spark", "version": "1.0.0"}


@app.get("/api/status")
def status():
    with LOCK:
        out = ACTIVE_OUTPUT
        job = dict(JOB)
    summary = read_json(out / "summary.json", {})
    metrics = read_json(out / "metrics.json", {})
    cameras = read_json(out / "cameras.json", [])
    incidents = read_json(out / "incidents.json", [])
    timeline = read_json(out / "timeline.json", [])
    if isinstance(cameras, dict): cameras = cameras.get("cameras", [])
    if isinstance(incidents, dict): incidents = incidents.get("incidents", [])
    if isinstance(timeline, dict): timeline = timeline.get("timeline", [])
    observations = read_json(out / "observations.json", [])
    benchmarks = read_json(ROOT / "reports/benchmark.json", read_json(ROOT / "outputs/benchmark.json", {}))
    metrics["benchmark"] = benchmarks
    metrics.setdefault("engine", "Apache Spark" if summary else "Awaiting Spark run")
    sources = read_json(ROOT / "data/sources.json", [])
    if isinstance(sources, dict): sources = sources.get("sources", [])
    for incident in incidents:
        if incident.get("evidence_path"):
            incident["evidence_url"] = _public_asset_url(incident["evidence_path"])
    modified = max((p.stat().st_mtime for p in out.glob("*.json")), default=None) if out.exists() else None
    return {
        "name": "Mine Safety Observatory",
        "mode": job["mode"], "running": job["running"], "engine": metrics.get("engine"),
        "last_updated": datetime.fromtimestamp(modified, timezone.utc).isoformat() if modified else None,
        "summary": summary, "metrics": metrics, "cameras": cameras, "incidents": incidents,
        "timeline": timeline, "observations": observations, "sources": sources, "assets": _assets(),
        "vision": read_json(ROOT / "outputs/vision/manifest.json", {}),
        "vision_analytics": read_json(ROOT / "outputs/vision_analytics/summary.json", {}),
        "job": job, "error": job["error"], "message": job["message"],
        "limitations": [
            "Prototype for human review; not a certified mine safety or machine shutdown system.",
            "Dashboard replay records are synthetic. Real public-image inference is shown separately.",
            "The pretrained PPE model has not been validated for deployment in underground mines.",
            "Unknown head visibility is excluded from helmet compliance, and displayed separately.",
            "Still images do not establish exposure duration or persistent incidents.",
            "Workers are camera-local tracks, not identified or deduplicated people across cameras.",
        ],
    }


def _launch(kind: str):
    global PROCESS, ACTIVE_OUTPUT
    with LOCK:
        if JOB["running"]:
            raise HTTPException(409, "A Spark job is already running. Stop it or wait for completion.")
        JOB.update(running=True, mode="synthetic replay" if kind == "stream" else "synthetic batch", kind=kind, error=None, message="Starting Apache Spark", started_at=datetime.now(timezone.utc).isoformat())
        STOP_REPLAY.clear()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_output = OUTPUT if kind == "batch" else ROOT / "outputs/runs" / run_id
    with LOCK:
        ACTIVE_OUTPUT = run_output
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    args = [sys.executable, "-m", "mine_safety", kind, "--output", str(run_output), "--master", os.environ.get("MINE_SAFETY_MASTER", "local[2]")]
    producer = None
    if kind == "batch":
        prepare_demo(ROOT / "data/demo")
        args += ["--input", str(ROOT / "data/demo/events.jsonl")]
    else:
        inbox = ROOT / "data/runtime" / run_id / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        args += ["--input", str(inbox), "--checkpoint", str(ROOT / "checkpoints" / run_id), "--trigger", "2"]
        producer = threading.Thread(target=replay, kwargs={"input_dir": inbox, "seconds": 120, "batch_seconds": 10, "delay": 4.0, "session_id": run_id, "stop_event": STOP_REPLAY}, daemon=True)
    env = dict(os.environ)
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    env.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    log_path = log_dir / f"{run_id}-{kind}.log"
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=(os.name != "nt"))
        with LOCK:
            PROCESS = process
            JOB["message"] = "Apache Spark is processing synthetic camera records"
        if producer: producer.start()
    except Exception as exc:
        log.close()
        with LOCK:
            JOB.update(running=False, error=str(exc), message="Could not start Spark")
        raise HTTPException(500, "Could not start Apache Spark. Check Java and project dependencies.") from exc

    def monitor():
        global PROCESS
        return_code = process.wait()
        log.close()
        STOP_REPLAY.set()
        with LOCK:
            if PROCESS is process:
                PROCESS = None
                stopping = JOB.get("stopping", False)
                JOB.update(running=False, message="Stopped" if stopping else "Analysis complete" if return_code == 0 else "Spark job failed", stopping=False)
                if return_code != 0 and not stopping:
                    JOB["error"] = f"Spark exited with code {return_code}. See {log_path.relative_to(ROOT)}."
    threading.Thread(target=monitor, daemon=True).start()
    return {"status": "started", "job_id": run_id, "kind": kind, "source_type": "synthetic"}


@app.post("/api/demo/start")
def start_demo():
    return _launch("stream")


class AnalysisRequest(BaseModel):
    input: str = "demo"


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    if request.input != "demo":
        raise HTTPException(400, "Use the CLI to analyse your own local input. Dashboard accepts the bundled demo only.")
    return _launch("batch")


def _stop_process():
    with LOCK:
        process = PROCESS
        STOP_REPLAY.set()
        if process and process.poll() is None:
            JOB["stopping"] = True
            JOB["message"] = "Stopping Spark and replay producer"
            try:
                if os.name != "nt": os.killpg(process.pid, signal.SIGTERM)
                else: process.terminate()
            except ProcessLookupError:
                pass
    return {"status": "stopping" if process else "idle"}


@app.post("/api/demo/stop")
def stop_demo():
    return _stop_process()


@app.get("/api/asset/{asset_path:path}")
def asset(asset_path: str):
    path = (ROOT / asset_path).resolve()
    if not _public_asset_url(path):
        raise HTTPException(404, "Asset not available")
    return FileResponse(path)


@app.get("/api/download/events")
def download_events():
    path = ROOT / "data/demo/events.jsonl"
    if not path.exists(): prepare_demo(path.parent)
    return FileResponse(path, filename="synthetic_camera_observations.jsonl", media_type="application/x-ndjson")


atexit.register(_stop_process)
