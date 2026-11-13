"""Real Apache Spark analytics for Mine Safety Spark.

The streaming dashboard sink deliberately favours transparent, restartable
course-project behaviour over an unbounded production serving architecture.
Every microbatch has a stable Parquet directory; snapshots are rebuilt with
Spark from the retained history. See docs/SPARK.md for the scaling boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OBSERVATION_COLUMNS = [
    "event_id", "event_time", "camera_id", "session_id", "track_id", "zone_id",
    "helmet_status", "in_restricted_zone", "confidence", "duration_s",
    "helmet_incident_id", "zone_incident_id", "evidence_path", "source_type",
    "duration_is_measured",
]

SAFETY_RULES = [
    ("helmet_missing", "medium", "Visible missing helmet", "rule-v1"),
    ("restricted_zone", "high", "Entry into a configured restricted zone", "rule-v1"),
]


def observation_schema():
    from pyspark.sql.types import (
        BooleanType, DoubleType, StringType, StructField, StructType,
    )

    numeric = {"confidence", "duration_s"}
    return StructType([
        StructField(
            name,
            BooleanType() if name in {"in_restricted_zone", "duration_is_measured"} else
            DoubleType() if name in numeric else StringType(),
            True,
        )
        for name in OBSERVATION_COLUMNS
    ] + [StructField("_corrupt_record", StringType(), True)])


def create_spark(master: str = "local[2]", app_name: str = "MineSafetyAnalytics"):
    from pyspark.sql import SparkSession

    loopback_cluster = master.startswith("spark://127.0.0.1:") or master.startswith("spark://localhost:")
    if master.startswith("local") or loopback_cluster:
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    builder = (
        SparkSession.builder.master(master).appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "false")
    )
    if master.startswith("spark://"):
        builder = (builder.config("spark.executor.memory", "512m")
                   .config("spark.executor.cores", "1")
                   .config("spark.cores.max", "2"))
        if loopback_cluster:
            builder = builder.config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def _valid_observations(raw):
    """Return a marked dataframe. Invalid rows remain countable, never hazards."""
    from pyspark.sql import functions as F

    df = (raw.withColumn("_event_ts", F.try_to_timestamp("event_time"))
          .withColumn("duration_is_measured", F.coalesce(
              F.col("duration_is_measured"), F.col("source_type") != "public_image",
          )))
    identifiers = ["event_id", "camera_id", "session_id", "track_id", "zone_id"]
    valid = F.col("_corrupt_record").isNull() & F.col("_event_ts").isNotNull()
    for name in identifiers:
        valid = valid & F.col(name).isNotNull() & (F.length(F.trim(F.col(name))) > 0)
    valid = valid & F.col("helmet_status").isin("present", "absent", "unknown")
    valid = valid & F.col("source_type").isin("synthetic", "public_image", "video")
    valid = valid & F.col("in_restricted_zone").isNotNull()
    valid = valid & F.col("confidence").between(0.0, 1.0) & ~F.isnan("confidence")
    valid = valid & (F.col("duration_s") > 0) & (F.col("duration_s") < 1e9)
    valid = valid & ~F.isnan("duration_s")
    helmet_id = F.col("helmet_incident_id").isNotNull() & (F.length(F.col("helmet_incident_id")) > 0)
    zone_id = F.col("zone_incident_id").isNotNull() & (F.length(F.col("zone_incident_id")) > 0)
    is_image = F.col("source_type") == "public_image"
    valid = valid & (is_image | (F.col("helmet_status") != "absent") | helmet_id)
    valid = valid & (is_image | ~F.col("in_restricted_zone") | zone_id)
    return df.withColumn("_valid", F.coalesce(valid, F.lit(False)))


def _deduplicate(df):
    """Resolve even conflicting repeated IDs deterministically."""
    from pyspark.sql import Window, functions as F

    payload_hash = F.sha2(F.to_json(F.struct(*OBSERVATION_COLUMNS)), 256)
    window = Window.partitionBy("event_id").orderBy("_event_ts", payload_hash)
    return (df.withColumn("_row_number", F.row_number().over(window))
            .where(F.col("_row_number") == 1)
            .select(*OBSERVATION_COLUMNS, "_event_ts"))


def _prepare(raw):
    from pyspark.sql import functions as F

    marked = _valid_observations(raw).persist()
    counts = marked.agg(
        F.count("*").alias("input_rows"),
        F.sum(F.col("_valid").cast("long")).alias("valid_rows"),
    ).first().asDict()
    counts = {key: int(value or 0) for key, value in counts.items()}
    counts["invalid_rows"] = counts["input_rows"] - counts["valid_rows"]
    clean = _deduplicate(marked.where("_valid")).persist()
    counts["unique_observations"] = clean.count()
    counts["duplicate_rows"] = counts["valid_rows"] - counts["unique_observations"]
    marked.unpersist()
    return clean, counts


def _incident_dataframe(clean, spark):
    from pyspark.sql import functions as F

    common = ["camera_id", "session_id", "track_id", "zone_id", "_event_ts",
              "duration_s", "evidence_path", "source_type"]
    temporal = clean.where((F.col("source_type") != "public_image") & F.col("duration_is_measured"))
    helmet = temporal.where(F.col("helmet_status") == "absent").select(
        F.col("helmet_incident_id").alias("incident_id"),
        F.lit("helmet_missing").alias("type"), *common,
    )
    zone = temporal.where("in_restricted_zone").select(
        F.col("zone_incident_id").alias("incident_id"),
        F.lit("restricted_zone").alias("type"), *common,
    )
    episodes = helmet.unionByName(zone).withColumn(
        "_end_ts",
        F.expr("timestamp_micros(unix_micros(_event_ts) + cast(duration_s * 1000000 as bigint))"),
    ).groupBy(
        "incident_id", "type", "camera_id", "session_id", "track_id", "zone_id", "source_type",
    ).agg(
        F.min("_event_ts").alias("_start_ts"),
        F.max("_end_ts").alias("_end_ts"),
        F.sum("duration_s").alias("duration_s"),
        F.max("evidence_path").alias("evidence_path"),
        F.count("*").alias("observations"),
    )
    rules = spark.createDataFrame(SAFETY_RULES, ["type", "severity", "rule_description", "rule_version"])
    return episodes.join(F.broadcast(rules), "type", "left")


def _ratio(numerator, denominator):
    from pyspark.sql import functions as F

    return F.when(denominator > 0, F.round(100.0 * numerator / denominator, 2))


def _json_rows(df):
    return [row.asDict(recursive=True) for row in df.collect()]


def _replace_parquet(df, path: Path) -> None:
    """Atomic directory switch for the local/shared-filesystem demo."""
    pending = path.with_name(f".{path.name}.pending")
    old = path.with_name(f".{path.name}.previous")
    if pending.exists():
        shutil.rmtree(pending)
    df.write.mode("overwrite").parquet(str(pending))
    if old.exists():
        shutil.rmtree(old)
    if path.exists():
        os.replace(path, old)
    os.replace(pending, path)
    if old.exists():
        shutil.rmtree(old)


def _write_analysis(clean, output: Path, spark, counts: dict, started: float,
                    mode: str = "batch", extra_metrics: dict | None = None,
                    write_history: bool = True) -> dict:
    from pyspark.sql import functions as F

    output.mkdir(parents=True, exist_ok=True)
    measured = (F.col("source_type") != "public_image") & F.col("duration_is_measured")
    present = F.when(measured & (F.col("helmet_status") == "present"), F.col("duration_s")).otherwise(0.0)
    absent = F.when(measured & (F.col("helmet_status") == "absent"), F.col("duration_s")).otherwise(0.0)
    unknown = F.when(measured & (F.col("helmet_status") == "unknown"), F.col("duration_s")).otherwise(0.0)
    restricted = F.when(measured & F.col("in_restricted_zone"), F.col("duration_s")).otherwise(0.0)
    observed = F.when(measured, F.col("duration_s")).otherwise(0.0)
    worker = F.struct("camera_id", "session_id", "track_id")
    basic = clean.agg(
        F.count("*").alias("observations"), F.countDistinct("camera_id").alias("cameras"),
        F.countDistinct(worker).alias("workers"), F.sum(present).alias("helmet_seconds"),
        F.sum(absent).alias("missing_helmet_seconds"), F.sum(unknown).alias("unknown_seconds"),
        F.sum(observed).alias("observed_seconds"), F.sum(restricted).alias("restricted_seconds"),
        F.sum((F.col("helmet_status") == "present").cast("long")).alias("helmet_present_observations"),
        F.sum((F.col("helmet_status") == "absent").cast("long")).alias("helmet_absent_observations"),
        F.sum((F.col("helmet_status") == "unknown").cast("long")).alias("helmet_unknown_observations"),
        F.sum((F.col("source_type") == "public_image").cast("long")).alias("public_image_observations"),
    ).first().asDict()
    basic = {key: (value if value is not None else 0) for key, value in basic.items()}
    incidents = _incident_dataframe(clean, spark).persist()
    incident_counts = {row["type"]: row["count"] for row in incidents.groupBy("type").count().collect()}
    assessable = basic["helmet_seconds"] + basic["missing_helmet_seconds"]
    summary = {
        **basic,
        "helmet_incidents": int(incident_counts.get("helmet_missing", 0)),
        "zone_incidents": int(incident_counts.get("restricted_zone", 0)),
        "high_priority_incidents": int(incident_counts.get("restricted_zone", 0)),
        "assessable_seconds": assessable,
        "compliance_pct": round(100 * basic["helmet_seconds"] / assessable, 2) if assessable else None,
        "unknown_pct": round(100 * basic["unknown_seconds"] / basic["observed_seconds"], 2)
        if basic["observed_seconds"] else None,
        "worker_definition": "distinct camera/session/track identities, not unique people",
        "last_updated": _utc_now(),
    }

    camera_episodes = incidents.groupBy("camera_id", "zone_id").agg(
        F.sum((F.col("type") == "helmet_missing").cast("long")).alias("helmet_incidents"),
        F.sum((F.col("type") == "restricted_zone").cast("long")).alias("zone_incidents"),
    )
    cameras = clean.groupBy("camera_id", "zone_id").agg(
        F.count("*").alias("observations"), F.sum(present).alias("helmet_seconds"),
        F.sum(absent).alias("missing_helmet_seconds"), F.sum(unknown).alias("unknown_seconds"),
        F.sum(observed).alias("observed_seconds"),
    ).join(camera_episodes, ["camera_id", "zone_id"], "left").fillna(
        0, subset=["helmet_incidents", "zone_incidents"],
    ).withColumn(
        "compliance_pct", _ratio(F.col("helmet_seconds"), F.col("helmet_seconds") + F.col("missing_helmet_seconds")),
    ).withColumn("unknown_pct", _ratio(F.col("unknown_seconds"), F.col("observed_seconds"))).withColumn(
        "status", F.when(F.col("zone_incidents") > 0, "high")
        .when(F.col("helmet_incidents") > 0, "warning")
        .when(F.col("compliance_pct").isNull(), "unknown").otherwise("clear"),
    ).orderBy("camera_id", "zone_id")

    time_observations = clean.groupBy(F.window("_event_ts", "1 minute")).agg(
        F.count("*").alias("observations"),
    )
    time_incidents = incidents.groupBy(F.window("_start_ts", "1 minute")).agg(
        F.sum((F.col("type") == "helmet_missing").cast("long")).alias("helmet_incidents"),
        F.sum((F.col("type") == "restricted_zone").cast("long")).alias("zone_incidents"),
    )
    timeline = (time_observations.join(time_incidents, "window", "left")
                .fillna(0, subset=["helmet_incidents", "zone_incidents"])
                .select(F.date_format("window.start", "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("time"),
                        "helmet_incidents", "zone_incidents", "observations")
                .orderBy("time"))
    incident_output = incidents.select(
        "incident_id", "type", "camera_id", "zone_id", "track_id", "session_id",
        F.date_format("_start_ts", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").alias("start_time"),
        F.date_format("_end_ts", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").alias("end_time"),
        "duration_s", "severity", "source_type", "evidence_path", "observations", "rule_version",
    ).orderBy(F.desc("start_time"), "incident_id").limit(500)
    source_counts = {row["source_type"]: row["count"]
                     for row in clean.groupBy("source_type").count().collect()}
    if write_history:
        _replace_parquet(clean.select(*OBSERVATION_COLUMNS), output / "events.parquet")
    _atomic_json(output / "cameras.json", _json_rows(cameras))
    _atomic_json(output / "timeline.json", _json_rows(timeline))
    _atomic_json(output / "incidents.json", _json_rows(incident_output))
    _atomic_json(output / "observations.json", _json_rows(
        clean.orderBy(F.desc("_event_ts"), "event_id").select(*OBSERVATION_COLUMNS).limit(500),
    ))
    metrics = {
        "engine": "Apache Spark", "spark_version": spark.version,
        "master": spark.sparkContext.master, "application_id": spark.sparkContext.applicationId,
        "execution_mode": "local threads" if spark.sparkContext.master.startswith("local") else "standalone cluster",
        "mode": mode, "runtime_seconds": round(time.perf_counter() - started, 3),
        **counts, "source_type_counts": source_counts, "last_updated": summary["last_updated"],
        "incident_display_limit": 500, "observation_display_limit": 500,
        "snapshot_semantics": "all retained observations; one row per event_id",
        "risk_semantics": "configured rule severity, not an accident probability",
        **(extra_metrics or {}),
    }
    if mode == "streaming":
        metrics["batch_duration_seconds"] = metrics["runtime_seconds"]
    _atomic_json(output / "metrics.json", metrics)
    # Commit marker last: API readers may use this timestamp to detect a refresh.
    _atomic_json(output / "summary.json", summary)
    incidents.unpersist()
    return summary


def run_batch(input_path, output_dir, master: str = "local[2]") -> dict:
    started = time.perf_counter()
    spark = create_spark(master)
    raw = spark.read.schema(observation_schema()).json(str(Path(input_path).resolve()))
    clean, counts = _prepare(raw)
    try:
        return _write_analysis(clean, Path(output_dir).resolve(), spark, counts, started)
    finally:
        clean.unpersist()


def run_stream(input_dir, output_dir, checkpoint_dir, master: str = "local[2]",
               trigger_seconds: int = 2, available_now: bool = False):
    """Start one real Structured Streaming query with an idempotent demo sink.

    Resume with the SAME input/output/checkpoint paths. Do not delete a
    checkpoint and reuse its history directory: batch IDs would be ambiguous.
    """
    spark = create_spark(master, "MineSafetyStreaming")
    incoming, output, checkpoint = (Path(p).resolve() for p in
                                     (input_dir, output_dir, checkpoint_dir))
    incoming.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    history = output / "stream_history"
    records = output / "stream_batches"
    history.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    identity = output / "stream_identity.json"
    identity_data = {"input_dir": str(incoming), "checkpoint_dir": str(checkpoint)}
    if identity.exists() and json.loads(identity.read_text()) != identity_data:
        raise ValueError("This output belongs to a different stream/checkpoint. Use a new output directory.")
    if any(history.glob("batch_*")) and not (checkpoint / "metadata").exists():
        raise ValueError("History exists but its checkpoint is missing. Restore it or use a fresh output directory.")
    _atomic_json(identity, identity_data)

    def write_batch(batch, batch_id):
        started = time.perf_counter()
        clean, batch_counts = _prepare(batch)
        try:
            target = history / f"batch_{batch_id:012d}"
            clean.select(*OBSERVATION_COLUMNS).write.mode("overwrite").parquet(str(target))
            _atomic_json(records / f"batch_{batch_id:012d}.json", batch_counts)
        finally:
            clean.unpersist()
        paths = [str(path) for path in sorted(history.glob("batch_*")) if (path / "_SUCCESS").exists()]
        if not paths:
            return
        from pyspark.sql import functions as F

        historic = spark.read.parquet(*paths).withColumn("_event_ts", F.to_timestamp("event_time"))
        all_clean = _deduplicate(historic).persist()
        batch_stats = [json.loads(path.read_text()) for path in records.glob("batch_*.json")]
        counts = {key: sum(item.get(key, 0) for item in batch_stats)
                  for key in ("input_rows", "valid_rows", "invalid_rows")}
        counts["unique_observations"] = all_clean.count()
        counts["duplicate_rows"] = counts["valid_rows"] - counts["unique_observations"]
        try:
            _write_analysis(all_clean, output, spark, counts, started, "streaming", {
                "batch_id": int(batch_id), "batches_processed": len(batch_stats),
                "batch_input_rows": batch_counts["input_rows"],
                "trigger_seconds": trigger_seconds, "checkpoint_dir": str(checkpoint),
                "serving_scope": "bounded demonstration; full-history snapshot per microbatch",
            })
        finally:
            all_clean.unpersist()

    raw = (spark.readStream.schema(observation_schema())
           .option("maxFilesPerTrigger", 10).json(str(incoming)))
    writer = (raw.writeStream.queryName("mine_safety_dashboard")
              .foreachBatch(write_batch).option("checkpointLocation", str(checkpoint)))
    writer = writer.trigger(availableNow=True) if available_now else writer.trigger(
        processingTime=f"{trigger_seconds} seconds",
    )
    query = writer.start()
    if available_now:
        query.awaitTermination()
    return query


def run_window_stream(input_dir, output_dir, checkpoint_dir, master: str = "local[2]",
                      available_now: bool = False, watermark: str = "30 seconds"):
    """Native stateful Spark windows, separate from the current dashboard sink.

    Append output is FINALIZED windows only. The event-time watermark must
    advance beyond a window; stopping a replay does not finalise every window.
    """
    from pyspark.sql import functions as F

    spark = create_spark(master, "MineSafetyEventTimeWindows")
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    raw = spark.readStream.schema(observation_schema()).json(str(Path(input_dir).resolve()))
    valid = (_valid_observations(raw).where("_valid")
             .where((F.col("source_type") != "public_image") & F.col("duration_is_measured"))
             .withWatermark("_event_ts", watermark))
    unique = valid.dropDuplicatesWithinWatermark(["event_id"])
    windows = unique.groupBy(F.window("_event_ts", "1 minute"), "camera_id", "zone_id", "source_type").agg(
        F.count("*").alias("observations"),
        F.sum(F.when(F.col("helmet_status") == "present", F.col("duration_s")).otherwise(0.0)).alias("helmet_seconds"),
        F.sum(F.when(F.col("helmet_status") == "absent", F.col("duration_s")).otherwise(0.0)).alias("missing_helmet_seconds"),
        F.sum(F.when(F.col("helmet_status") == "unknown", F.col("duration_s")).otherwise(0.0)).alias("unknown_seconds"),
        F.sum(F.when(F.col("in_restricted_zone"), F.col("duration_s")).otherwise(0.0)).alias("restricted_seconds"),
    )
    writer = (windows.writeStream.format("parquet").outputMode("append")
              .option("path", str(Path(output_dir).resolve()))
              .option("checkpointLocation", str(Path(checkpoint_dir).resolve())))
    writer = writer.trigger(availableNow=True) if available_now else writer.trigger(processingTime="2 seconds")
    query = writer.start()
    if available_now:
        query.awaitTermination()
    return query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["batch", "stream", "windows"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="data/checkpoints/main")
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--available-now", action="store_true")
    args = parser.parse_args()
    if args.mode == "batch":
        print(json.dumps(run_batch(args.input, args.output, args.master), indent=2))
    else:
        fn = run_stream if args.mode == "stream" else run_window_stream
        query = fn(args.input, args.output, args.checkpoint, master=args.master,
                   available_now=args.available_now)
        if not args.available_now:
            query.awaitTermination()


if __name__ == "__main__":
    main()
