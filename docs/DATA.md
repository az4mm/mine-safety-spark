# Computer vision, public data and evaluation

## What is implemented

The project performs actual CPU object detection using two public, pretrained
ONNX models. A COCO YOLOv8n model finds people. A PPE YOLOv8n model finds hardhats
and explicit `NO-Hardhat` objects. A conservative spatial association rule maps
each head-region detection to at most one person. A helmet carried near the waist
does not count as worn. No matching evidence, or conflicting evidence, produces
`unknown` rather than an assumed violation.

OpenCV reads images and video. ONNX Runtime performs inference. The project then
produces JSONL observations for real Apache Spark processing. Model inference
runs outside Spark; Spark performs distributed analytics on those observations.
No synthetic detector is substituted when weights or inference fail.

**This project reuses pretrained weights. It does not claim to have trained a
mine-specific detector.** A genuine underground photograph is included in the
inference demonstration, alongside construction images. That demonstration is
not a mine-domain accuracy benchmark.

## Reproduce the public-image experiment

Run these commands from the project directory after installing the documented
dependencies:

```bash
python scripts/fetch_model.py
python scripts/fetch_public_data.py --labelled-sample
python scripts/run_public_vision.py --include-labelled-sample
python -m mine_safety.vision_evaluation
```

The model download verifies pinned SHA256 hashes and file sizes. Photo downloads
verify hashes. The fixed labelled sample checks its row IDs, annotations and image
hashes against `data/HARDHAT_SAMPLE_MANIFEST.json` before accepting the download.
No login, payment, API key or GPU is needed for these downloads and CPU inference.
Network access to the documented public hosts is required on a fresh checkout.

Outputs:

| File | Meaning |
|---|---|
| `outputs/vision/observations.jsonl` | Actual model-derived person observations, suitable for Spark input |
| `outputs/vision/manifest.json` | Model details, source attribution, per-image results and annotated gallery assets |
| `outputs/vision/<image-id>/vision_report.json` | Raw detections, confidence scores, coordinates and measured inference latency |
| `outputs/vision/<image-id>/*_frame_000000.jpg` | Prediction overlay on the source photo |
| `outputs/vision/evaluation.json` | Explicit fixed-sample matching results, including every prediction and reference box |

The public-image input is separate from the synthetic streaming input. For a
direct Spark batch call:

```python
from mine_safety.spark_jobs import run_batch

run_batch(
    "outputs/vision/observations.jsonl",
    "outputs/public_image_analytics",
    master="local[2]",
)
```

Still photographs have no measured time interval. Their required `duration_s=1`
is a nominal observation weight, with `duration_is_measured=false`. They are
excluded from measured exposure-duration and temporal-incident KPIs. Their
helmet status counts are model predictions. `helmet_incident_id` and
`zone_incident_id` remain null because a still frame cannot establish persistent
episodes. Processing timestamps are marked as assumed rather than original
capture times.

## Model provenance

| Component | Source | License | Role |
|---|---|---|---|
| PPE detector | [SafetyVision YOLOv8n v1](https://huggingface.co/ayushgupta7777/safetyvision-yolov8) | AGPL-3.0 | Hardhat and explicit no-hardhat detections |
| Person detector | [Kalray YOLOv8n ONNX export](https://huggingface.co/Kalray/yolov8) | AGPL-3.0 | COCO person detections |

Exact model revisions, hashes, file sizes, class mappings and links are recorded
in `models/MANIFEST.json` and `models/PERSON_MODEL_MANIFEST.json`.
The PPE export includes class names in ONNX metadata. The person export does not;
its class names were read from the publisher's
[classes file](https://github.com/kalray/kann-model-zoo/blob/main/networks/object-detection/yolov8/onnx/classes.txt).
External class mappings are accepted only for the pinned model hash. The project
implements its own standard RGB letterboxing, normalization and class-aware NMS;
it does not include the publisher's preprocessing source code.

Both networks take a 640 × 640 RGB float tensor scaled to [0, 1]. Confidence is
set to 0.35, with class-aware non-maximum suppression at IoU 0.45. The association
rule only considers the upper 45% of each person box and assigns each candidate
to one person. These are declared prototype choices, not validated mine safety
thresholds.

The public model's upstream benchmark numbers are not presented as this project's
results. An exploratory check of the same author's larger v2 model was not
selected; the reproducible final pipeline uses the pinned v1 model and the
separate person detector.

## Photographs and attribution

The machine-readable source register is `data/PUBLIC_DATA_MANIFEST.json`.

| Image | Credit and source | License | Domain |
|---|---|---|---|
| Oilshale mine 01 | [Peeter paaver, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Oilshale_mine_01.jpg) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Actual underground oil shale mine in Estonia |
| Construction Workers | [Paul Keleher, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Construction_Workers.jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | Construction; not mining |
| Construction workers in Iran 02 | [Mostafa Meraji / Mostafameraji, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Construction_workers_in_Iran_02.jpg) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | Difficult construction silhouette; not mining |

The downloaded files are Wikimedia thumbnails. The project adds prediction boxes
and text to produce the annotated derivatives. The authors do not endorse this
project. Predictions do not establish that any depicted person violated a rule.
The dark silhouette is deliberately retained even when detection fails.

The main gallery contains these three photographs plus a fixed twenty-image
public sample. On the recorded run, all 23 images produced **73 person
observations: 50 helmet-present, 5 model-predicted helmet-absent, and 18 unknown**.
The raw outputs contained 57 hardhat detections and 8 no-hardhat detections.
Raw item counts and person-assigned status counts differ because some item
detections cannot be assigned unambiguously to a detected person.

On the underground photograph the person detector found four people, while the
PPE detector found one hardhat: the resulting statuses were one present and three
unknown. This is a useful demonstration of domain limitations, rather than a
claim that the other three miners lacked helmets.

## Fixed public labelled sample

The sample uses rows 0–19 from `full/test` of
[keremberke/hard-hat-detection](https://huggingface.co/datasets/keremberke/hard-hat-detection).
The dataset card credits [Roboflow Universe Projects' Hard Hats v2](https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/2)
and declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit: Roboflow Universe Projects; Hugging Face conversion by keremberke.
Images are stored as 640 × 640 examples, and reference boxes use COCO xywh
coordinates. The sample selection was fixed before inference; failed cases are
not removed.

This is a small descriptive check. **Overlap with the pretrained model's
upstream training data is unknown. It is not a verified independent holdout and
is not mine-domain validation.** The two reference classes are `hardhat` and
`no-hardhat`; there are no person-association or incident labels.

At confidence ≥ 0.35, same-class IoU ≥ 0.50 and one-to-one confidence-ordered
matching, the recorded run produced:

| Class | True positive | False positive | False negative | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Hardhat | 44 | 11 | 21 | 0.8000 | 0.6769 | 0.7333 |
| No-hardhat | 3 | 5 | 9 | 0.3750 | 0.2500 | 0.3000 |
| Micro total | 47 | 16 | 30 | 0.7460 | 0.6104 | 0.6714 |

Precision is TP/(TP+FP); recall is TP/(TP+FN). These values describe localization
at one declared confidence threshold. **mAP is not measured.** In particular,
the low no-hardhat recall is a concrete limitation of this baseline. Do not
reinterpret these numbers as accident-prevention performance.

## Video mode and episode semantics

```bash
python -m mine_safety.vision your_video.mp4 --output outputs/my_video --camera-id camera-01 --zone-id loading-area --sample-fps 2 --max-frames 120 --polygon '[[0.5,0.5],[1,0.5],[1,1],[0.5,1]]'
```

Polygon coordinates are normalized [0, 1] coordinates on a fixed camera image.
The approximate ground point is the bottom centre of the person box. A box cut
off at the image bottom has an unassessable ground point and cannot establish a
zone intrusion. Pixel geometry does not measure metres or true machine clearance.

The tracker uses greedy one-to-one IoU matching, with an IoU threshold of 0.25 and
a 1.5-second maximum detection gap. This is a transparent baseline, not a
validated re-identification system. Track IDs are scoped by camera and session.
A positive helmet-absence episode requires one second of consecutive explicit
absence evidence. An unknown observation never emits an active helmet incident.
A short unknown gap can preserve an already confirmed episode's ID; a clear
helmet-present observation closes it. Re-entry starts a new ID. Zone episodes
are confirmed on configured-region entry and retain an ID during uninterrupted
occupancy.

`duration_s` in video mode represents the media interval assigned to a sampled
observation. It is a sampling-based approximation of exposure, not a claim of
continuous visual observation between sampled frames. If a capture start time is
known, pass `--start-time 2026-11-13T10:00:00+00:00`; otherwise the run uses its
processing start plus the media offset and marks the timestamp as assumed.

### Video IO smoke check

```bash
python scripts/smoke_video.py
```

This script repeats the licensed underground photo into a twelve-frame MP4 at
three frames per second. It is **synthetic temporal data**, not a real mining
video. The recorded check decoded all 12 frames, produced 48 person observations,
retained four track IDs, and retained four illustrative zone-episode IDs. It
checks video IO, sampling and repeated-frame identity persistence only; it does
not assess motion tracking accuracy. Results live separately in
`outputs/vision_video_smoke/summary.json` and never enter the public-image gallery
or synthetic multi-camera demonstration automatically.

## Next evaluation with actual mining data

[DsDPM-66](https://github.com/znrry/DsDPM-66) is a public underground drilling dataset
with `coal_miner` and `mining_helmet` task subsets, described in
[Scientific Data](https://www.nature.com/articles/s41597-025-05118-1).
It was researched for the project plan but **not downloaded, trained on or
evaluated in this delivered baseline**. A future mine-domain evaluation should
hold out complete scenes, audit missing-helmet negatives and complete any missing
joint annotations before combining task folders. Adjacent frames from the same
scene must not be randomly split between training and final test sets.

The current software can ingest real videos and already produces the observation
schema Spark needs. A future deployment claim requires representative footage,
independent ground truth, false-alert and missed-event measurements, and tested
camera/zone configuration. None of those results are invented here.
