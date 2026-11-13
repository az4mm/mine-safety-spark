"""Check a public checkout and existing results without rerunning Spark/inference.

The optional result checks inspect existing artifacts. Passing them establishes
file integrity and internal consistency, not a new accuracy or runtime test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
from typing import Any, Callable
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def required_files(paths: list[str]) -> str:
    missing = [path for path in paths if not (ROOT / path).is_file()]
    require(not missing, "Missing files: " + ", ".join(missing))
    empty = [path for path in paths if (ROOT / path).stat().st_size == 0]
    require(not empty, "Empty files: " + ", ".join(empty))
    return f"{len(paths)} nonempty files"


def check_sources() -> str:
    return required_files([
        "README.md", "LICENSE", "pyproject.toml", "requirements.txt",
        "mine_safety/cli.py", "mine_safety/server.py", "mine_safety/spark_jobs.py",
        "mine_safety/vision.py", "mine_safety/tracking.py",
        "mine_safety/vision_evaluation.py", "mine_safety/simulation.py",
        "scripts/fetch_model.py", "scripts/fetch_public_data.py",
        "scripts/run_public_vision.py", "scripts/smoke_video.py",
        "web/index.html", "web/app.js", "web/style.css",
        "docs/QUICKSTART.md", "docs/DATA.md", "docs/SPARK.md",
        "data/demo/events.jsonl", "data/demo/expected.json",
        "data/PUBLIC_DATA_MANIFEST.json", "data/HARDHAT_SAMPLE_MANIFEST.json",
        "models/MANIFEST.json", "models/PERSON_MODEL_MANIFEST.json",
        "tests/test_simulation.py", "tests/test_server.py",
        "tests/test_spark.py", "tests/test_vision.py",
    ])


def check_manifests() -> str:
    for name in ("models/MANIFEST.json", "models/PERSON_MODEL_MANIFEST.json"):
        manifest = read_json(name)
        require(len(manifest["revision"]) == 40, f"Unpinned model revision: {name}")
        require(re.fullmatch(r"[a-f0-9]{64}", manifest["sha256"]) is not None,
                f"Invalid model hash: {name}")
        require(manifest["revision"] in manifest["download_url"],
                f"Model download URL does not pin its revision: {name}")
        require(bool(manifest.get("license")), f"Model license missing: {name}")
    read_json("data/PUBLIC_DATA_MANIFEST.json")
    read_json("data/HARDHAT_SAMPLE_MANIFEST.json")
    return "two pinned model manifests and two parseable data manifests"


def check_runtime() -> str:
    require((3, 10) <= sys.version_info[:2] < (3, 14),
            "This package requires Python 3.10–3.13; Python 3.12 was tested.")
    packages = ["pyspark", "fastapi", "uvicorn", "numpy", "opencv-python-headless",
                "onnxruntime", "Pillow"]
    installed: dict[str, str] = {}
    for name in packages:
        try:
            installed[name] = version(name)
        except PackageNotFoundError as error:
            raise ValueError(f"Missing runtime dependency {name}; install requirements.txt") from error
    for name, expected in {"pyspark": "4.0.1", "onnxruntime": "1.23.2",
                           "opencv-python-headless": "4.12.0.88"}.items():
        require(installed[name] == expected,
                f"{name} is {installed[name]}, expected {expected}")
    result = subprocess.run(["java", "-version"], capture_output=True, text=True,
                            check=True, timeout=10)
    java_output = result.stdout + result.stderr
    match = re.search(r'version\s+"(\d+)', java_output)
    require(match is not None and int(match.group(1)) >= 17,
            "Java 17+ must be available on PATH")
    return f"Python {sys.version.split()[0]}, Java {match.group(1)}, " + ", ".join(
        f"{name} {installed[name]}" for name in packages
    )


def check_model_files() -> str:
    total = 0
    for name in ("models/MANIFEST.json", "models/PERSON_MODEL_MANIFEST.json"):
        manifest = read_json(name)
        path = ROOT / "models" / manifest["filename"]
        require(path.is_file(), f"Download models first: python scripts/fetch_model.py ({path.name})")
        require(path.stat().st_size == manifest["bytes"], f"Model size mismatch: {path.name}")
        with path.open("rb") as handle:
            digest = (hashlib.file_digest(handle, "sha256").hexdigest()
                      if sys.version_info >= (3, 11)
                      else hashlib.sha256(handle.read()).hexdigest())
        require(digest == manifest["sha256"], f"Model SHA256 mismatch: {path.name}")
        total += path.stat().st_size
    return f"two model sizes and SHA256 hashes match; {total:,} bytes"


def check_public_results() -> str:
    manifest = read_json("outputs/vision/manifest.json")
    rows = [json.loads(line) for line in (ROOT / "outputs/vision/observations.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), "No public-image observations found; run scripts/run_public_vision.py")
    require(len({row["event_id"] for row in rows}) == len(rows), "Duplicate public-image event IDs")
    require(all(row["source_type"] == "public_image" for row in rows),
            "Synthetic/video records are mixed into the public-image observations")
    require(all(row.get("duration_is_measured") is False for row in rows),
            "A still image has been marked as measured duration")
    require(all(not row.get("helmet_incident_id") and not row.get("zone_incident_id") for row in rows),
            "Still-image rows must not establish temporal incident IDs")
    counts = Counter(row["helmet_status"] for row in rows)
    require(manifest["person_observations"] == len(rows), "Manifest observation count mismatch")
    require(dict(counts) == manifest["helmet_status_counts"], "Manifest helmet count mismatch")
    summary = read_json("outputs/vision_analytics/summary.json")
    metrics = read_json("outputs/vision_analytics/metrics.json")
    require(metrics["engine"] == "Apache Spark" and metrics["spark_version"] == "4.0.1",
            "Public-image analytics must record Apache Spark 4.0.1")
    require(metrics["input_rows"] == metrics["valid_rows"] ==
            metrics["unique_observations"] == len(rows) and
            metrics["invalid_rows"] == metrics["duplicate_rows"] == 0,
            "Public-image Spark input, validity or unique-observation counts do not match")
    require(summary["observations"] == len(rows), "Spark has a different public-image observation count")
    require(metrics["source_type_counts"] == {"public_image": len(rows)},
            "Spark public-image result contains another source type")
    for status in ("present", "absent", "unknown"):
        require(summary[f"helmet_{status}_observations"] == counts[status],
                f"Spark helmet-{status} count mismatch")
    require(summary["helmet_incidents"] == summary["zone_incidents"] == 0,
            "Still-image analytics must not report temporal incidents")
    require(summary["observed_seconds"] == summary["assessable_seconds"] == 0,
            "Still-image analytics must not report measured exposure")
    require(summary["compliance_pct"] is None, "Image-only time compliance must be null")
    parquet = ROOT / "outputs/vision_analytics/events.parquet"
    require((parquet / "_SUCCESS").is_file() and bool(list(parquet.glob("*.parquet"))),
            "Spark public-image Parquet output is incomplete")
    return f"{len(rows)} actual public-image observations match Spark; zero time/incident KPIs"


def check_cv_evaluation() -> str:
    evaluation = read_json("outputs/vision/evaluation.json")
    require(evaluation["images_evaluated"] > 0, "Evaluation contains no images")
    classes = evaluation["per_class"]
    micro = evaluation["micro"]
    for key in ("tp", "fp", "fn"):
        require(sum(item[key] for item in classes.values()) == micro[key],
                f"Evaluation micro {key} does not equal class totals")
    for label, item in list(classes.items()) + [("micro", micro)]:
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        require(abs(item["precision"] - precision) < 1e-8 and
                abs(item["recall"] - recall) < 1e-8,
                f"Evaluation precision/recall arithmetic mismatch: {label}")
    require(evaluation["mAP"] is None, "This experiment did not compute mAP")
    return f"{evaluation['images_evaluated']} labelled sample images; recorded metric arithmetic is consistent"


def check_demo_results() -> str:
    expected = read_json("data/demo/expected.json")
    # Recorded experiment evidence survives future dashboard replays.
    location = ("reports/results/demo/summary.json" if
                (ROOT / "reports/results/demo/summary.json").is_file() else
                "outputs/current/summary.json")
    summary = read_json(location)
    for key in ("observations", "cameras", "workers", "helmet_incidents", "zone_incidents",
                "assessable_seconds", "helmet_seconds", "missing_helmet_seconds",
                "unknown_seconds", "restricted_seconds"):
        require(summary[key] == expected[key], f"Synthetic result mismatch: {key}")
    return f"{location}: scenario totals match the synthetic reference"


def check_benchmark() -> str:
    location = "reports/benchmark.json" if (ROOT / "reports/benchmark.json").is_file() else "outputs/benchmark.json"
    benchmark = read_json(location)
    results = benchmark["results"]
    require(bool(results), "Benchmark has no recorded results")
    require("Apache Spark" in benchmark["kind"],
            "Benchmark does not identify the Apache Spark engine")
    for item in results:
        require(item["spark_version"] == "4.0.1",
                "Benchmark result must record Spark 4.0.1")
        require(item["input_rows"] > 0 and
                item["aggregated_observations"] == item["input_rows"],
                "Benchmark aggregation did not preserve input counts")
        require(0 < item["output_groups"] <= item["input_rows"],
                "Benchmark output-group count is invalid")
        require(item["repeats"] > 0 and len(item["run_seconds"]) == item["repeats"] and
                all(math.isfinite(value) and value > 0 for value in item["run_seconds"]),
                "Benchmark repetitions/timings are incomplete")
        seconds = item["median_seconds"]
        require(math.isfinite(seconds) and seconds > 0 and
                math.isclose(seconds, median(item["run_seconds"]), abs_tol=0.0001),
                "Benchmark median does not match its recorded repetitions")
        expected_rate = item["input_rows"] / seconds
        # Timings are rounded to four decimals; rates use the unrounded median.
        rate_tolerance = max(0.1, expected_rate * 0.0001 / seconds)
        require(math.isfinite(item["rows_per_second"]) and
                abs(item["rows_per_second"] - expected_rate) <= rate_tolerance,
                "Benchmark throughput is inconsistent with input rows and median time")
    return f"{len(results)} recorded Spark configurations; repetitions and aggregation counts consistent"


def check_recorded_tests() -> str:
    required_files(["reports/test_results.xml", "reports/end_to_end.json"])
    report = ET.parse(ROOT / "reports/test_results.xml").getroot()
    cases = list(report.iter("testcase"))
    require(bool(cases), "Recorded test report contains no test cases")
    require(not list(report.iter("failure")) and not list(report.iter("error")),
            "Recorded test report contains a failure or error")
    integration = read_json("reports/end_to_end.json")
    require(integration.get("passed") is True, "Recorded API/streaming integration did not pass")
    require(integration["metrics"]["engine"] == "Apache Spark" and
            integration["metrics"]["spark_version"] == "4.0.1",
            "Recorded integration must identify Apache Spark 4.0.1")
    expected = read_json("data/demo/expected.json")
    for key in ("observations", "helmet_incidents", "zone_incidents", "assessable_seconds"):
        require(integration["summary"][key] == expected[key],
                f"Recorded integration scenario mismatch: {key}")
    return f"{len(cases)} recorded test cases without failures/errors; integration totals match the synthetic reference"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="check installed dependencies and Java, without starting Spark")
    parser.add_argument("--results", action="store_true", help="check downloaded models and reproduced result consistency")
    args = parser.parse_args()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("source files", check_sources), ("source manifests", check_manifests)
    ]
    if args.runtime:
        checks.append(("runtime", check_runtime))
    if args.results:
        checks.extend([
            ("downloaded models", check_model_files),
            ("public-image Spark results", check_public_results),
            ("CV sample evaluation", check_cv_evaluation),
            ("synthetic scenario", check_demo_results),
            ("recorded Spark benchmark", check_benchmark),
            ("recorded test evidence", check_recorded_tests),
        ])
    failures = 0
    for label, check in checks:
        try:
            print(f"PASS {label}: {check()}")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
                ET.ParseError) as error:
            failures += 1
            print(f"FAIL {label}: {error}")
    print(f"{len(checks) - failures}/{len(checks)} checks passed. No Spark job, model inference or browser test was run.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
