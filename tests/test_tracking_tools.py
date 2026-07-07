"""Tests des briques M3 : trajectoire de pan, stats de tracking, payload.

Tout est pur Python — exécutable dans la CI sans dataset ni libs lourdes.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from make_test_video import pan_positions  # noqa: E402
from track_video import build_tracking_payload  # noqa: E402

from weedtrack.pipeline import TrackingStats  # noqa: E402


# ---------- pan_positions ----------

def test_pan_couvre_toute_la_largeur():
    positions = pan_positions(1280, 720, 640, steps=100)
    assert len(positions) == 100
    assert positions[0] == (0, 40)          # y centré : (720-640)//2
    assert positions[-1] == (640, 40)       # bord droit : 1280-640
    xs = [x for x, _ in positions]
    assert xs == sorted(xs)                 # monotone gauche → droite


def test_pan_image_trop_petite():
    with pytest.raises(ValueError):
        pan_positions(600, 720, 640, steps=10)


def test_pan_steps_invalide():
    with pytest.raises(ValueError):
        pan_positions(1280, 720, 640, steps=1)


# ---------- TrackingStats ----------

def test_tracking_stats_ids_uniques():
    stats = TrackingStats()
    for frame in range(3):
        stats.start_frame()
        stats.add("weed", 1)       # même piste vue sur 3 frames
        stats.add("weed", 2 + frame)  # nouvelle piste à chaque frame
    stats.add("crop", 10)
    stats.add("crop", None)        # détection non encore confirmée

    summary = stats.summary()
    assert summary["frames"] == 3
    assert summary["detections"] == 8
    assert summary["unique_tracks"] == {"crop": 1, "weed": 4}
    assert summary["untracked_detections"] == 1
    assert summary["detections_per_frame"] == 2.67


def test_tracking_stats_vide():
    summary = TrackingStats().summary()
    assert summary["frames"] == 0
    assert summary["detections_per_frame"] == 0.0
    assert summary["unique_tracks"] == {}


# ---------- build_tracking_payload ----------

def test_payload_tracking():
    payload = build_tracking_payload(
        summary={"frames": 600, "pipeline_fps": 18.5, "unique_tracks": {"weed": 12}},
        source="test_video.mp4",
        tracker="bytetrack.yaml",
        device="mps",
        imgsz=640,
        conf=0.25,
        weights="best.pt",
    )
    assert payload["results"]["frames"] == 600
    assert payload["tracker"] == "bytetrack.yaml"
    assert payload["source"] == "test_video.mp4"
    assert "generated_at" in payload
