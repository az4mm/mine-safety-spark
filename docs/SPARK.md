# Apache Spark implementation and measured evaluation

## What runs in Spark

This project uses **PySpark 4.0.1**. There is no pandas replacement and no
simulated Spark execution. The vision component produces observations; Spark
validates them, resolves repeated observation IDs, joins an explicit safety-rule
table, creates violation episodes, computes exposure-weighted summaries and
time windows, and writes a Parquet history and dashboard snapshots.

`metrics.json` identifies the Spark version, master URL, application ID, input
and rejected counts, runtime, and source provenance. Synthetic replay is
labelled `source_type=synthetic` throughout; it is not evidence of CV accuracy.

### Batch pipeline

```bash
.venv/bin/python -m mine_safety.spark_jobs batch \
  --input data/demo/events.jsonl --output outputs/demo --master 'local[2]'
```

1. Read JSON with an explicit schema and parse timestamps in UTC.
2. Reject malformed timestamps, invalid enums, missing identifiers, bad
   confidence/duration values, and temporal violations without episode IDs.
3. Deduplicate `event_id` with a deterministic ordering. Identical records count
   once. If repeated IDs disagree, the earliest event time and then a stable
   payload hash select one row; the application should prevent conflicting IDs
   at ingestion rather than treating corrections as new facts.
4. Group observations by stable episode ID, camera, session, track and zone.
5. Broadcast-join the versioned safety-rule dimension: missing helmet is medium
   priority; restricted-zone entry is high priority. These are demonstration
   rules, not a calibrated accident probability or an official risk assessment.
6. Compute overall, per-camera/zone, and one-minute summaries with DataFrames.
7. Write Parquet and JSON snapshots. Only bounded samples of the latest 500
   observations and incidents are collected for the UI; totals use all rows.

The output directory contains `summary.json`, `metrics.json`, `cameras.json`,
`incidents.json`, `timeline.json`, `observations.json`, and `events.parquet`.
An observation is not an incident: multiple frames with the same episode ID
contribute duration to one episode. One-minute incident charts count episode
**starts**, so a long incident is not counted again in every minute.

### How the metrics are defined

| Metric | Definition |
|---|---|
| `observations` | Valid, unique observation IDs, including still images |
| `workers` | Distinct camera/session/track identities, not identified people |
| `helmet_seconds` | Observed time with a visible helmet |
| `missing_helmet_seconds` | Observed time with a visible uncovered head |
| `assessable_seconds` | Helmet-present plus helmet-absent observed time |
| `compliance_pct` | 100 × helmet-present time / assessable time |
| `unknown_pct` | 100 × unassessable time / all measured observed time |
| `restricted_seconds` | Summed tracked-worker time inside the configured zone |
| `helmet_incidents` | Distinct grouped temporal missing-helmet episodes |
| `zone_incidents` | Distinct grouped temporal zone-entry episodes |

Public still images use nominal sample weights, not measured durations. Rows
with `source_type=public_image` or `duration_is_measured=false` do not contribute
exposure seconds or temporal incident counts. Their image-level detections are
reported as `helmet_present_observations`, `helmet_absent_observations`,
`helmet_unknown_observations`, and `public_image_observations`. No assessable
temporal data means `compliance_pct=null`, not 100% compliance.

Track observations must represent non-overlapping sample durations. Summing
overlapping estimates would overstate exposure. Camera status summarises
**recorded history**; it does not certify that a current scene is safe.

## Live dashboard query

```bash
.venv/bin/python -m mine_safety.spark_jobs stream \
  --input data/stream_inbox --output outputs/current \
  --checkpoint data/checkpoints/current --master 'local[2]'
```

`readStream` watches completed JSONL files. Write a file outside the watched
directory and atomically move it into place. Never append to an already
processed input file. Each trigger consumes up to ten new files. A two-second
trigger is a scheduling target; a batch that takes longer delays the next
batch. The dashboard is near-real-time course-project monitoring, not a
hard-real-time protective controller.

The `foreachBatch` sink writes each microbatch to the deterministic
`stream_history/batch_<batch_id>` directory. Re-executing a batch overwrites the
same history partition. Spark then deduplicates event IDs over retained
history and recomputes the summary. Statistics use one file per batch ID as
well. The bounded UI snapshots are replaced atomically, and `summary.json`
is written last as a refresh marker.

**Recovery:** restart with the same input, output and checkpoint paths. A
completed file will not normally be reprocessed. A failed batch may replay;
its stable path and the global ID deduplication prevent duplicated totals.
New input files containing old observation IDs also do not increase counts.
The code rejects an existing history paired with a missing or different
checkpoint. To start a completely new run, use new input/output/checkpoint
directories. Preserve the checkpoint and history together.

This is a transparent **bounded demonstration sink**. Recomputing retained
history on every microbatch has growing cost. It is not an architecture claim
for unlimited camera traffic. A production extension would use incremental
stateful episode processing and an idempotent serving database. The separate
native window query below demonstrates incremental state management without
misrepresenting the dashboard implementation.

The directory rename strategy assumes one shared/local filesystem visible to
the driver and executors. A separate-machine deployment needs shared storage
or a redesign for an object store; driver-local files alone are insufficient.

## Native event-time windows and watermarks

```bash
.venv/bin/python -m mine_safety.spark_jobs windows \
  --input data/stream_inbox --output outputs/finalized_windows \
  --checkpoint data/checkpoints/windows --available-now
```

This is a separate real Structured Streaming query. It uses:

- Parsed event time, rather than file arrival time.
- `withWatermark(..., '30 seconds')` to bound late-data state.
- `dropDuplicatesWithinWatermark` for repeated observation IDs.
- Native one-minute window aggregates grouped by camera, zone and provenance.
- An append-mode Parquet sink and persistent query checkpoint.

Append mode emits **finalized** windows only. A watermark advances with
observed event times, not simply with wall-clock waiting. A replay ending at
00:02 may leave its newest window unfinalized. This is expected, and these
finalized windows are not used as the current-alert view. Data arriving
behind a closed window is excluded from this streaming aggregate; batch
history analytics can still include retained records according to its
different completeness policy. Run a new query/checkpoint when changing
stateful grouping or other incompatible query settings.

## Local execution and a standalone cluster

`local[1]` and `local[2]` mean one and two Spark worker **threads in a local
process**, respectively. They are useful for development and a parallelism
comparison; they do not establish multi-machine distribution.

The supplied Linux scripts launch one master and two standalone worker
processes without Docker:

```bash
bash scripts/start_cluster.sh
.venv/bin/python -m mine_safety.spark_jobs batch \
  --input data/demo/events.jsonl --output outputs/cluster_batch \
  --master spark://127.0.0.1:7077
bash scripts/stop_cluster.sh
```

The master UI is `http://127.0.0.1:8080`; the worker UIs use ports 8081/8082.
Each worker advertises one core and 1 GiB; the application requests two
one-core, 512 MiB executors. The driver runs in client mode. All processes
share one physical host, so this demonstrates cluster scheduling and
executors, not independent hardware or proven horizontal scaling. The
scripts bind services to loopback. Stop only this project's daemons using
the matching script.

## Benchmarks and tests

```bash
.venv/bin/python -m pytest tests/test_spark.py -q
.venv/bin/python -m mine_safety.benchmark \
  --masters 'local[1]' 'local[2]' --sizes 10000 100000 --repeats 3 \
  --output outputs/benchmark.json
```

The benchmark starts a fresh Python/JVM process for each master and size,
generates deterministic synthetic records, caches the input, warms the query,
and measures repeated broadcast-join and one-minute aggregation executions.
It verifies that output observation totals match the generated input.
The JSON contains every repetition, median duration, input count, partition
count, Spark version, application ID and measured throughput. Spark startup,
generation and warm-up are separately recorded and excluded from the timed
query. Result collection is included. This is an analytics compute benchmark,
not end-to-end video throughput or safety-detection accuracy. Small workloads
may run slower with more parallelism; retain those results honestly.

Tests verify validation, global deduplication, unknown treatment, still-image
exclusion, duration weighting, episode counts, rule joins, restart recovery,
duplicate records across microbatches, native watermarks and late events.

## Official references

- [Spark 4.0.1 documentation](https://spark.apache.org/docs/4.0.1/)
- [Structured Streaming](https://spark.apache.org/docs/4.0.1/streaming/index.html)
- [Structured Streaming DataFrame APIs](https://spark.apache.org/docs/4.0.1/streaming/apis-on-dataframes-and-datasets.html)
- [Submitting applications and local master definitions](https://spark.apache.org/docs/4.0.1/submitting-applications.html)
- [Spark standalone deployment](https://spark.apache.org/docs/4.0.1/spark-standalone.html)

These documents describe engine semantics; measured project results are the
JSON artifacts generated by the actual runs, not documentation performance
claims.
