"""Fetch documented public photographs and an optional fixed labelled sample."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "MineSafetySpark/0.1 educational research (public course project)"


def download(url, destination, expected_sha=None, maximum_bytes=20_000_000):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_sha and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha:
        return expected_sha
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > maximum_bytes:
                        raise ValueError("Public sample exceeded size limit")
                    digest.update(block)
                    target.write(block)
        actual = digest.hexdigest()
        if expected_sha and actual != expected_sha:
            raise ValueError(f"Checksum changed for {destination.name}; review the source before updating the manifest")
        os.replace(temporary, destination)
        return actual
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def fetch_photographs(output=ROOT/"data/public"):
    manifest = json.loads((ROOT/"data/PUBLIC_DATA_MANIFEST.json").read_text())
    for asset in manifest["assets"]:
        destination = Path(output)/asset["filename"]
        download(asset["download_url"], destination, asset["sha256"])
        print(f"Verified public photograph: {destination}")


def fetch_labelled_sample(output=ROOT/"data/public/hardhat_test_sample", count=20):
    if not 1 <= count <= 20:
        raise ValueError("This pinned smoke sample supports between 1 and 20 rows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    public_manifest = json.loads((ROOT/"data/PUBLIC_DATA_MANIFEST.json").read_text())["labelled_sample"]
    snapshot = json.loads((ROOT/"data/HARDHAT_SAMPLE_MANIFEST.json").read_text())
    expected = {r["row_index"]: r for r in snapshot["records"]}
    params = urllib.parse.urlencode({"dataset": public_manifest["dataset"], "config": "full", "split": "test", "offset": 0, "length": count})
    request = urllib.request.Request("https://datasets-server.huggingface.co/rows?"+params, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    records = []
    for wrapper in data["rows"]:
        row = wrapper["row"]
        filename = f"test_{wrapper['row_idx']:04d}.jpg"
        # The signed CDN URL is obtained from the public dataset response. It is
        # not committed because it expires. The row index and checksum are saved.
        reference = expected[wrapper["row_idx"]]
        if row["image_id"] != reference["image_id"] or row["objects"]["bbox"] != reference["boxes_xywh"] or row["objects"]["category"] != reference["categories"]:
            raise ValueError("Public dataset row changed; review the pinned annotations before updating")
        checksum = download(row["image"]["src"], output/filename, reference["sha256"])
        records.append({"row_index": wrapper["row_idx"], "image_id": row["image_id"],
                        "file": filename, "width": row["width"], "height": row["height"],
                        "sha256": checksum, "boxes_xywh": row["objects"]["bbox"],
                        "categories": row["objects"]["category"]})
        print(f"Downloaded labelled test row {wrapper['row_idx']}: {filename}")
    metadata = dict(public_manifest, configuration="full", split="test", sample_size=len(records),
                    class_names=["hardhat", "no-hardhat"], records=records)
    (output/"annotations.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output/"annotations.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled-sample", action="store_true", help="Also fetch the fixed 20-image labelled sample")
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    fetch_photographs()
    if args.labelled_sample:
        fetch_labelled_sample(count=args.sample_count)
