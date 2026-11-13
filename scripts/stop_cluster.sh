#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MINE_PYTHON:-${project_dir}/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="python3"; fi
spark_dir="$($python_bin -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).parent)')"
run_dir="${MINE_CLUSTER_DIR:-${project_dir}/outputs/cluster}"
export SPARK_PID_DIR="$run_dir/pids"
export SPARK_LOG_DIR="$run_dir/logs"
for identity in mine-worker1 mine-worker2; do
    SPARK_IDENT_STRING="$identity" "$spark_dir/sbin/spark-daemon.sh" stop org.apache.spark.deploy.worker.Worker 1
done
SPARK_IDENT_STRING=mine-master "$spark_dir/sbin/spark-daemon.sh" stop org.apache.spark.deploy.master.Master 1
rm -f "$run_dir/cluster.env"
echo "Stopped this project's Spark standalone daemons."
