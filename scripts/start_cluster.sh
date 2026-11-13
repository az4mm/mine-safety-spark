#!/usr/bin/env bash
set -euo pipefail

# One physical host, one standalone master, two worker processes. No Docker.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MINE_PYTHON:-${project_dir}/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="python3"; fi
spark_dir="$($python_bin -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).parent)')"
run_dir="${MINE_CLUSTER_DIR:-${project_dir}/outputs/cluster}"
mkdir -p "$run_dir/pids" "$run_dir/logs" "$run_dir/work1" "$run_dir/work2"
export SPARK_PID_DIR="$run_dir/pids"
export SPARK_LOG_DIR="$run_dir/logs"
export SPARK_LOCAL_IP="127.0.0.1"
export SPARK_DAEMON_MEMORY="256m"
master_port="${MINE_MASTER_PORT:-7077}"
master_ui="${MINE_MASTER_UI_PORT:-8080}"
master_url="spark://127.0.0.1:${master_port}"
if [[ -f "$run_dir/cluster.env" ]]; then
    echo "Cluster metadata already exists: $run_dir/cluster.env. Run stop_cluster.sh first."
    exit 1
fi
SPARK_IDENT_STRING=mine-master "$spark_dir/sbin/spark-daemon.sh" start \
    org.apache.spark.deploy.master.Master 1 --host 127.0.0.1 --port "$master_port" --webui-port "$master_ui"
SPARK_IDENT_STRING=mine-worker1 "$spark_dir/sbin/spark-daemon.sh" start \
    org.apache.spark.deploy.worker.Worker 1 --host 127.0.0.1 --port 7081 --webui-port 8081 \
    --cores 1 --memory 1024m --work-dir "$run_dir/work1" "$master_url"
SPARK_IDENT_STRING=mine-worker2 "$spark_dir/sbin/spark-daemon.sh" start \
    org.apache.spark.deploy.worker.Worker 1 --host 127.0.0.1 --port 7082 --webui-port 8082 \
    --cores 1 --memory 1024m --work-dir "$run_dir/work2" "$master_url"
cat > "$run_dir/cluster.env" <<EOF
MINE_SPARK_DIR=$spark_dir
MINE_MASTER_URL=$master_url
MINE_MASTER_UI=http://127.0.0.1:$master_ui
EOF
echo "Standalone master: $master_url"
echo "Master UI: http://127.0.0.1:$master_ui"
echo "Topology: one physical host, two one-core worker processes."
echo "Workers may need a few seconds to register."
