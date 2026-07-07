"""Tests de la mise en forme des métriques (fonction pure, sans ML)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate import build_metrics_payload  # noqa: E402


def make_payload(**overrides):
    base = dict(
        map50=0.71234,
        map50_95=0.45678,
        per_class={
            "crop": {"ap50": 0.6011, "ap50_95": 0.4022},
            "weed": {"ap50": 0.8233, "ap50_95": 0.5144},
        },
        fps=14.567,
        fps_images=100,
        device="mps",
        imgsz=640,
        weights="best.pt",
        ultralytics_version="8.4.90",
        machine="Darwin arm64",
    )
    base.update(overrides)
    return build_metrics_payload(**base)


def test_arrondis_et_champs():
    payload = make_payload()
    assert payload["map50"] == 0.7123
    assert payload["map50_95"] == 0.4568
    assert payload["per_class"]["weed"]["ap50"] == 0.8233
    assert payload["inference"]["fps"] == 14.57
    assert payload["inference"]["device"] == "mps"
    assert payload["weights"] == "best.pt"
    assert "generated_at" in payload
    assert "CC BY 4.0" in payload["dataset"]


def test_fps_absent_tolere():
    payload = make_payload(fps=None, fps_images=0)
    assert payload["inference"]["fps"] is None
    assert payload["inference"]["benchmark_images"] == 0
