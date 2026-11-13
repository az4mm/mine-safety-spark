"""Check MP4 decoding/tracking on a repeated public photo, not a motion dataset."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_smoke(output=ROOT/"outputs/vision_video_smoke"):
    import cv2
    from mine_safety.vision import analyze_source
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(ROOT/"data/public/oilshale_mine_01.jpg"))
    if photo is None:
        raise FileNotFoundError("Run scripts/fetch_public_data.py first")
    # An even resolution avoids codec-specific rounding of 853-pixel height.
    photo = cv2.resize(photo, (960, 640))
    fixture = output/"synthetic_repeated_image.mp4"
    writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*"mp4v"), 3, (960, 640))
    if not writer.isOpened():
        raise RuntimeError("The MP4 encoder is unavailable")
    for _ in range(12):
        writer.write(photo)
    writer.release()
    report = analyze_source(fixture, output/"results", camera_id="synthetic-video-io-camera",
                            session_id="repeated-public-image-smoke", sample_fps=3, max_frames=12,
                            zone_id="illustrative-entire-frame", polygon=[[0,0], [1,0], [1,1], [0,1]])
    # This is generated temporal data, so it must never be presented as measured
    # worker motion or real footage. Keep it out of the real public-image input.
    rows = [row for frame in report["frames"] for row in frame["observations"]]
    for row in rows:
        row["source_type"] = "synthetic"
        row["fixture_type"] = "repeated_public_still_image_video"
    for frame in report["frames"]:
        frame["source_type"] = "synthetic"
    report["source_type"] = "synthetic"
    report["fixture_type"] = "repeated_public_still_image_video"
    report["limitations"].insert(0, "Repeated-image MP4 fixture only. Exercises video IO and ID persistence; does not evaluate motion, tracking accuracy, or mine safety.")
    for asset in report["assets"]:
        asset["source_type"] = "synthetic"
        asset["description"] = "Repeated-image MP4 fixture; no real motion is represented."
    with (output/"results/observations.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row)+"\n")
    (output/"results/vision_report.json").write_text(json.dumps(report, indent=2))
    track_ids = {r["track_id"] for r in rows}
    zone_ids = {r["zone_incident_id"] for r in rows if r["zone_incident_id"]}
    assert report["frames_processed"] == 12, "Video sampling did not decode all frames"
    assert rows and track_ids, "Video fixture produced no person observations"
    assert len(zone_ids) <= len(track_ids), "Repeated frames incorrectly created repeated zone incidents"
    summary = {"passed": True, "fixture_type": "synthetic repeated public photo MP4",
               "frames_encoded": 12, "frames_processed": report["frames_processed"],
               "fps": 3, "person_observations": len(rows), "unique_tracks": len(track_ids),
               "zone_incident_ids": len(zone_ids), "source_type": "synthetic",
               "notes": "Tests IO/tracking stability only; no real motion or tracking accuracy claim.",
               "attribution": "Oilshale mine 01 by Peeter paaver, CC BY 4.0; resized and repeated in an MP4."}
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/vision_video_smoke")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))
