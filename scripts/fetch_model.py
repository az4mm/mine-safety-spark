"""Download the pinned public model; verify bytes before replacing a local file."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch_model(destination=None, manifest_path=None):
    manifest = json.loads((Path(manifest_path) if manifest_path else ROOT/"models/MANIFEST.json").read_text())
    destination = Path(destination) if destination else ROOT/"models"/manifest["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == manifest["sha256"]:
        print(f"Verified existing model: {destination}")
        return destination
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            request = urllib.request.Request(manifest["download_url"], headers={"User-Agent": "MineSafetySpark/0.1 educational research"})
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > manifest["bytes"]:
                        raise ValueError("Downloaded model exceeds the pinned size")
                    digest.update(block)
                    target.write(block)
        if size != manifest["bytes"] or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("Model checksum/size mismatch; refusing unverified weights")
        os.replace(temporary, destination)
        print(f"Downloaded and SHA256 verified {size:,} bytes: {destination}")
        return destination
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ppe-only", action="store_true", help="Skip the auxiliary COCO person model")
    args = parser.parse_args()
    fetch_model(args.output)
    if not args.ppe_only:
        fetch_model(manifest_path=ROOT/"models/PERSON_MODEL_MANIFEST.json")
