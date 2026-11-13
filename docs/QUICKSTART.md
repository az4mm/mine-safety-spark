# Run the project on your computer

## Runtime overview

Follow the Linux/WSL2 instructions below to run Apache Spark and computer
vision. Native Windows commands are also provided. The dashboard is served at
`http://127.0.0.1:8000` after starting the Python application.

The completed runtime was tested on Linux, Python 3.12 and Java 21. The supplied
package supports Python 3.10–3.13; Python 3.12 is the reproducible choice used
here. PySpark 4.0.1 needs Java 17 or later, as documented by
[Apache Spark](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html).
The project uses CPU inference and does not require a GPU, paid service, Docker,
Kaggle login or API key. A fresh setup needs network access to PyPI and the public
model/data hosts.

## A. Linux or Ubuntu in WSL2

### 1. Open the project folder

Clone the repository, extract a source archive, or open an existing checkout,
then enter its `mine-safety-spark` folder in a terminal. The following commands
assume that this folder is the current directory.

On Ubuntu 24.04, if Python/Java are not installed:

```bash
sudo apt update
sudo apt install python3 python3-venv openjdk-21-jre-headless
```

Check the installed tools:

```bash
python3 --version
java -version
```

### 2. Create the environment and install the project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file installs the local project using the single dependency
definition in `pyproject.toml`, including PySpark 4.0.1, ONNX Runtime 1.23.2 and
OpenCV 4.12. Python's active virtual environment is important: use it consistently
for the server, Spark driver and model scripts.

If Spark cannot resolve the local machine name, set this in the same terminal:

```bash
export SPARK_LOCAL_IP=127.0.0.1
```

If Java cannot be discovered, set `JAVA_HOME` to the actual installed Java
directory. For example, on an x86-64 Ubuntu system with the package above this
is commonly `/usr/lib/jvm/java-21-openjdk-amd64`; verify the directory on your
computer before using it.

### 3. Run the bundled synthetic Spark demo

```bash
python -m mine_safety generate
python -m mine_safety batch --input data/demo/events.jsonl --output outputs/current --master "local[2]"
python -m mine_safety serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Keep the
terminal running. Press Ctrl+C in that terminal to stop the server.

Expected scenario totals: 720 unique synthetic observations, three cameras,
six camera-local tracks, five missing-helmet episodes and four zone episodes.
These values come from a known simulation, not detected mine incidents. The
first JVM launch can be slower than later dashboard refreshes.

The dashboard's replay button launches real Structured Streaming and a
synthetic file producer. Wait for its job status to show activity. Use its stop
button to stop the running replay. A running stream and a new batch are not
launched together through the dashboard.

## B. Windows: recommended WSL2 route

This follows the tested Linux runtime while keeping Windows as your desktop.
[Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install)
documents supported Windows versions and the installation steps.

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if Windows requests it. Launch Ubuntu and complete its initial user
setup. Then follow section A inside the Ubuntu terminal. If WSL is already
installed, check it with:

```powershell
wsl --list --verbose
```

Keep the working copy in the Ubuntu home filesystem when practical. Extract
or copy the package into a folder such as `~/projects/mine-safety-spark`.
Do not reuse a virtual environment created by Windows inside Ubuntu; create
`.venv` again in Ubuntu. After starting the server, open
`http://127.0.0.1:8000` in your Windows browser.

## C. Native Windows PowerShell

These are concrete native commands, but native Windows execution was not part
of this project's completed verification. The Linux cluster helper scripts
require Linux/WSL2. Windows Spark setups can also need additional Hadoop-native
configuration; if a native filesystem/`winutils.exe`/`HADOOP_HOME` error blocks
Spark, use section B to reproduce the tested environment.

Install Python 3.12 and a 64-bit Java 21 JDK. Configure `JAVA_HOME` to your actual
JDK installation directory and ensure Java is available on `PATH`. Open a new
PowerShell window and verify:

```powershell
py -3.12 --version
java -version
```

Clone or extract the project to a regular local directory, for example
`C:\Projects\mine-safety-spark`, and run:

```powershell
cd C:\Projects\mine-safety-spark
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYSPARK_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:SPARK_LOCAL_IP = "127.0.0.1"
.\.venv\Scripts\python.exe -m mine_safety generate
.\.venv\Scripts\python.exe -m mine_safety batch --input data/demo/events.jsonl --output outputs/current --master "local[2]"
.\.venv\Scripts\python.exe -m mine_safety serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The commands invoke the
virtual-environment interpreter directly, so no PowerShell activation-policy
change is needed.

For the public-image workflow, use the same interpreter:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_model.py
.\.venv\Scripts\python.exe scripts/fetch_public_data.py --labelled-sample
.\.venv\Scripts\python.exe scripts/run_public_vision.py --include-labelled-sample
.\.venv\Scripts\python.exe -m mine_safety batch --input outputs/vision/observations.jsonl --output outputs/vision_analytics --master "local[2]"
.\.venv\Scripts\python.exe -m mine_safety.vision_evaluation
```

The remaining examples use `python`; on native Windows replace it with
`.\.venv\Scripts\python.exe` when the environment is not activated.

## D. Reproduce all public-image results

From the project directory in the active environment:

```bash
python scripts/fetch_model.py
python scripts/fetch_public_data.py --labelled-sample
python scripts/run_public_vision.py --include-labelled-sample
python -m mine_safety batch --input outputs/vision/observations.jsonl --output outputs/vision_analytics --master "local[2]"
python -m mine_safety.vision_evaluation
```

The downloads comprise two public pretrained models totalling about 25 MB, three
public photographs and a small labelled sample. Both model hashes and the sample
annotations are checked against the supplied manifests. The final input includes
a genuine underground photograph plus construction/web images. There is no
training step: the package runs the documented pretrained weights.

Expected recorded observations: 73 persons, comprising 50 present, five absent
and 18 unknown helmet statuses. These are model outputs, not guaranteed ground
truth. The actual output files and confidence thresholds are inspectable.

The Spark public-image summary is written to **`outputs/vision_analytics`**,
separate from the synthetic dashboard data in `outputs/current`. Still images
do not contribute measured exposure time or persistent incident IDs. Therefore,
no time-based compliance percentage is expected for the image-only dataset.

Outputs to inspect:

| Path | Content |
|---|---|
| `outputs/vision/manifest.json` | Public image/model provenance and measured prediction counts |
| `outputs/vision/*/*.jpg` | Annotated model predictions |
| `outputs/vision/observations.jsonl` | Actual model observations sent to Spark |
| `outputs/vision_analytics/summary.json` | Spark's separate public-image summary |
| `outputs/vision_analytics/metrics.json` | Spark version/application ID/input validation statistics |
| `outputs/vision_analytics/events.parquet/` | Spark's persisted public-image history |
| `outputs/vision/evaluation.json` | Fixed-sample precision/recall and raw prediction/reference matching |

[DATA.md](DATA.md) explains every source, metric and limitation. The measured
sample is small and its overlap with the model's upstream training images is
unknown; it is not a mine-domain accuracy benchmark.

## E. Video IO smoke check and your own footage

```bash
python scripts/smoke_video.py
```

The check creates a twelve-frame MP4 by repeating a licensed public photograph.
Its rows are explicitly synthetic. The recorded run decoded all twelve frames
and preserved four tracks and four illustrative zone-episode IDs. This checks
video plumbing; it does not assess motion accuracy.

For an actual local video:

```bash
python -m mine_safety.vision your_video.mp4 --output outputs/my_video --camera-id camera-01 --zone-id loading-area --sample-fps 2 --max-frames 120 --polygon "[[0.5,0.5],[1,0.5],[1,1],[0.5,1]]"
python -m mine_safety batch --input outputs/my_video/observations.jsonl --output outputs/my_video_analytics
```

The polygon is a normalized image region configured by you. It is not an
automatically identified danger zone or a measurement in metres. Keep the
camera fixed. Use your own capture start timestamp with `--start-time` when it
is known; otherwise the run marks its event timestamps as assumed.

## F. Manual Structured Streaming

Open two terminals, each in the project directory with the same environment
activated. Use a fresh set of directories for a new demonstration, or keep an
existing input/output/checkpoint set together when resuming it.

Terminal 1:

```bash
python -m mine_safety stream --input data/runtime/manual01/inbox --output outputs/manual01 --checkpoint checkpoints/manual01 --trigger 2
```

Terminal 2:

```bash
python -m mine_safety replay --output data/runtime/manual01/inbox --seconds 120 --batch-seconds 10 --delay 4
```

The producer writes completed files atomically. Spark consumes them, stores
batch history, deduplicates IDs and refreshes its summaries. Read
`outputs/manual01/summary.json`, or start a dashboard in another terminal:

```bash
python -m mine_safety serve --output outputs/manual01 --port 8001
```

Open `http://127.0.0.1:8001`. Stop the stream with Ctrl+C when done. The producer
finishes after its configured data is replayed. The two-second trigger is a
scheduling target, not a guaranteed end-to-end latency.

See [SPARK.md](SPARK.md) for native event-time windows, watermarks, late data,
checkpoint recovery and the limits of the bounded demonstration sink.

## G. Two-worker Spark cluster and benchmarking

On Linux/WSL2:

```bash
bash scripts/start_cluster.sh
python -m mine_safety batch --input data/demo/events.jsonl --output outputs/cluster_batch --master spark://127.0.0.1:7077
bash scripts/stop_cluster.sh
```

The master UI uses `http://127.0.0.1:8080`. Both workers share the same host;
this demonstrates cluster processes and scheduling, not separate-machine
scaling. Stop the project's cluster after the experiment.

To reproduce the local parallelism benchmark:

```bash
python -m mine_safety.benchmark --masters "local[1]" "local[2]" --sizes 10000 100000 1000000 --repeats 3 --output outputs/benchmark.json
```

This starts several real Spark applications and takes longer than the unit
checks. Run it while the machine is otherwise idle. Its timed section is a
warmed analytics query, not full video processing.

## H. Tests and project verification

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_simulation.py tests/test_server.py tests/test_vision.py -q
python -m pytest tests/test_spark.py -q
python scripts/verify_project.py --runtime
```

The first test command needs no JVM. The second exercises real Spark, including
streaming behaviour. The verifier checks source files, manifests and the
environment; it does not run new Spark workloads.

On a fresh source checkout before downloading models or generating results,
run `python scripts/verify_project.py` for a source-file check. Add
`--runtime` to check Python/Java/dependencies, `--results` after reproducing the
experiments. The result checks validate downloaded model hashes and the
consistency of generated outputs. Each missing item is reported explicitly.

## Troubleshooting

| Symptom | Action |
|---|---|
| `java` is not found / Java gateway exits | Install Java 17 or 21, verify `java -version`, and correct `JAVA_HOME` to the actual installation directory. |
| Spark tries another Python interpreter | Use the same virtual environment. On Windows set `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` as shown above. |
| Native Windows filesystem / Hadoop helper error | Use the WSL2 route to reproduce the tested Linux environment. |
| Port 8000 is busy | Start `python -m mine_safety serve --port 8001` and open that port. |
| Missing model or public image | Run the fetch scripts; they verify hashes and do not silently substitute predictions. |
| Model/data host is blocked on your network | Permit the documented public hosts in your environment or download the pinned asset through an approved connection. The bundled synthetic Spark demo does not require model or image downloads. |
| A changed checksum is rejected | Do not disable verification. Check the documented upstream revision and update the manifest only after reviewing the new file. |
| Stream history/checkpoint mismatch | Resume with the original matching paths, or use new input/output/checkpoint directories for a new run. |
| Image-only compliance shows no percentage | Expected: still photos do not measure time exposure. Inspect helmet observation counts instead. |
| Low no-hardhat recall | This is a measured limitation of the pretrained baseline. Review the evaluation instead of claiming reliable mine deployment. |

Generated experiment data and checkpoints can be rebuilt. Keep any footage you
supply in a separate local directory outside the public repository.
