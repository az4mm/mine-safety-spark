"""Measured Spark SQL compute benchmarks on explicitly synthetic event rows.

This measures analytics computation, not camera inference, input transport, or
mine-safety accuracy. Every master runs in a fresh Python/JVM process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def measure(master: str, size: int, repeats: int = 3) -> dict:
    from pyspark.sql import functions as F
    from mine_safety.spark_jobs import create_spark

    boot = time.perf_counter()
    spark = create_spark(master, "MineSafetySyntheticAnalyticsBenchmark")
    startup_s = time.perf_counter() - boot
    prepared = time.perf_counter()
    events = spark.range(size, numPartitions=8).select(
        "id",
        F.concat(F.lit("CAM-"), (F.col("id") % 12).cast("string")).alias("camera_id"),
        F.concat(F.lit("ZONE-"), (F.col("id") % 4).cast("string")).alias("zone_id"),
        F.timestamp_seconds(F.lit(1767225600) + F.floor(F.col("id") / 12)).alias("event_time"),
        F.when(F.col("id") % 10 == 0, "absent")
        .when(F.col("id") % 10 == 1, "unknown").otherwise("present").alias("helmet_status"),
        (F.col("id") % 17 == 0).alias("in_restricted_zone"),
        F.lit(1.0).alias("duration_s"),
    ).cache()
    assert events.count() == size
    prepare_s = time.perf_counter() - prepared
    rules = spark.createDataFrame(
        [("ZONE-0", "loading"), ("ZONE-1", "haul road"),
         ("ZONE-2", "walkway"), ("ZONE-3", "work area")],
        ["zone_id", "zone_label"],
    )

    def query():
        return events.join(F.broadcast(rules), "zone_id").groupBy(
            "camera_id", "zone_id", "zone_label", F.window("event_time", "1 minute"),
        ).agg(
            F.count("*").alias("observations"),
            F.sum((F.col("helmet_status") == "present").cast("long")).alias("helmet_present"),
            F.sum((F.col("helmet_status") == "absent").cast("long")).alias("helmet_absent"),
            F.sum((F.col("helmet_status") == "unknown").cast("long")).alias("helmet_unknown"),
            F.sum(F.col("in_restricted_zone").cast("long")).alias("restricted"),
        )

    warmup = time.perf_counter()
    first = query().collect()
    warmup_s = time.perf_counter() - warmup
    assert sum(row["observations"] for row in first) == size
    durations = []
    final = []
    for _ in range(repeats):
        start = time.perf_counter()
        final = query().collect()
        durations.append(time.perf_counter() - start)
    total = sum(row["observations"] for row in final)
    if total != size:
        raise AssertionError(f"Benchmark aggregation lost observations: {total} != {size}")
    # Published benchmark evidence needs the execution mode, not the address or
    # ephemeral identity of the process that ran it.
    public_master = "standalone" if master.startswith("spark://") else master
    if "://" in public_master:
        public_master = public_master.partition("://")[0] + " cluster"
    result = {
        "master": public_master, "spark_version": spark.version,
        "source_type": "synthetic", "input_rows": size, "output_groups": len(final),
        "aggregated_observations": total, "repeats": repeats,
        "startup_seconds": round(startup_s, 4), "prepare_seconds": round(prepare_s, 4),
        "warmup_seconds": round(warmup_s, 4),
        "run_seconds": [round(value, 4) for value in durations],
        "median_seconds": round(statistics.median(durations), 4),
        "rows_per_second": round(size / statistics.median(durations), 1),
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
        "input_partitions": events.rdd.getNumPartitions(),
        "benchmark_scope": "cached synthetic records; SQL broadcast join and one-minute window aggregates; includes result collection",
        "excludes": ["model inference", "video decoding", "input transport", "Spark startup", "input generation"],
    }
    events.unpersist()
    spark.stop()
    return result


def run_benchmarks(output_path, masters=None, sizes=None, repeats=3) -> dict:
    masters = masters or ["local[1]", "local[2]"]
    sizes = sizes or [10000, 100000]
    results = []
    for master in masters:
        for size in sizes:
            command = [sys.executable, "-m", "mine_safety.benchmark", "--worker",
                       "--master", master, "--size", str(size), "--repeats", str(repeats)]
            child = subprocess.run(command, capture_output=True, text=True, check=True)
            lines = [line for line in child.stdout.splitlines() if line.startswith("BENCHMARK_JSON=")]
            if not lines:
                raise RuntimeError("Benchmark child returned no measured result")
            results.append(json.loads(lines[-1].split("=", 1)[1]))
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "kind": "measured Apache Spark analytics benchmark",
        "data_provenance": "deterministically generated synthetic observations; no accuracy claims",
        "topology_note": "local[N] is N threads on one host; standalone workers on the same host share physical resources",
        "publication_note": "Hostnames, network addresses and application identifiers are omitted; measured values are retained.",
        "results": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/benchmark.json")
    parser.add_argument("--masters", nargs="+", default=["local[1]", "local[2]"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--master", default="local[2]", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, default=10000, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print("BENCHMARK_JSON=" + json.dumps(measure(args.master, args.size, args.repeats)))
    else:
        result = run_benchmarks(args.output, args.masters, args.sizes, args.repeats)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
