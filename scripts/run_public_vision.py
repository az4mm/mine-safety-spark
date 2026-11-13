"""Produce actual public-image predictions, a combined Spark input, and gallery metadata."""
from pathlib import Path
import argparse
from collections import Counter
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mine_safety.vision import analyze_source


def run_public_vision(output=ROOT/"outputs/vision", include_labelled_sample=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    public_manifest = json.loads((ROOT/"data/PUBLIC_DATA_MANIFEST.json").read_text())
    inputs = [(ROOT/"data/public"/a["filename"], a) for a in public_manifest["assets"]]
    if include_labelled_sample:
        sample_path = ROOT/"data/public/hardhat_test_sample/annotations.json"
        sample = json.loads(sample_path.read_text())
        for row in sample["records"]:
            inputs.append((sample_path.parent/row["file"],
                           {"id": f"hardhat_{row['row_index']:04d}", "title": f"Public hardhat test row {row['row_index']}",
                            "source": sample["source"], "author": "Roboflow Universe Projects / keremberke",
                            "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "domain": "construction/web dataset, not mining"}))
    results, observations, assets = [], [], []
    for source, attribution in inputs:
        result = analyze_source(source, output/attribution["id"],
                                camera_id=f"public-{attribution['id']}",
                                session_id=f"public-image-{attribution['id']}")
        rows = [row for frame in result["frames"] for row in frame["observations"]]
        observations.extend(rows)
        results.append({"id": attribution["id"], "title": attribution["title"],
                        "source": attribution["source"], "domain": attribution["domain"],
                        "person_observations": result["person_observations"],
                        "detection_counts": result["detection_counts"],
                        "helmet_status_counts": result["helmet_status_counts"],
                        "inference_latency_ms": result["inference_latency_ms"],
                        "report_path": str((output/attribution["id"]/"vision_report.json").relative_to(ROOT))})
        for asset in result["assets"]:
            assets.append(dict(asset, url="/assets/"+asset["path"], title=attribution["title"],
                               source_url=attribution["source"], author=attribution["author"],
                               license=attribution["license"], license_url=attribution["license_url"],
                               domain=attribution["domain"],
                               description=f"Actual ONNX inference on a public photograph. {attribution['domain']}. Model predictions, not verified incidents."))
        print(f"{attribution['id']}: {result['detection_counts']} | statuses {result['helmet_status_counts']}", flush=True)
    with (output/"observations.jsonl").open("w", encoding="utf-8") as f:
        for row in observations:
            f.write(json.dumps(row)+"\n")
    manifest = {
        "source_type": "public_image", "engine": "ONNX Runtime CPU",
        "images_processed": len(inputs), "person_observations": len(observations),
        "helmet_status_counts": dict(Counter(r["helmet_status"] for r in observations)),
        "detection_counts": dict(sum((Counter(r["detection_counts"]) for r in results), Counter())),
        "observations_path": str((output/"observations.jsonl").relative_to(ROOT)),
        "assets": assets, "results": results,
        "models": [json.loads((ROOT/"models/MANIFEST.json").read_text()),
                   json.loads((ROOT/"models/PERSON_MODEL_MANIFEST.json").read_text())],
        "accuracy_measured_on_gallery": False,
        "limitations": ["Static predictions do not measure exposure duration or establish persistent incidents.",
                        "Public photo selection is an illustrative demonstration, not an accuracy benchmark.",
                        "Industrial pretrained model used without mine-specific fine-tuning.",
                        "A separate labelled-sample check, if present, is small and upstream training overlap is unknown."]}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/vision")
    parser.add_argument("--include-labelled-sample", action="store_true")
    args = parser.parse_args()
    manifest = run_public_vision(args.output, args.include_labelled_sample)
    print(f"Wrote {manifest['person_observations']} real-model observations and {manifest['images_processed']} annotated images")
