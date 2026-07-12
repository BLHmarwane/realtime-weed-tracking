"""Smoke tests : vérifient la santé du squelette sans dépendances lourdes.

Ils doivent passer sur n'importe quelle machine avec pytest, même sans
ultralytics/opencv installés — c'est la garantie que le repo reste clonable et
vérifiable en 30 secondes.
"""

import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_importable_sans_dependances_lourdes():
    import weedtrack
    from weedtrack import config, detect, pipeline, track  # noqa: F401

    assert weedtrack.__version__


def test_structure_du_projet():
    # Contrat minimal des fichiers nécessaires dans une copie publique.
    attendus = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "configs/config.yaml",
        "scripts/download_dataset.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "app/streamlit_app.py",
        "docker/Dockerfile",
        "data/README.md",
        "models/README.md",
    ]
    manquants = [rel for rel in attendus if not (PROJECT_ROOT / rel).exists()]
    assert not manquants, f"Fichiers manquants : {manquants}"


def test_config_yaml_valide():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text())
    assert config["model"]["weights"]
    assert config["model"]["imgsz"] > 0
    assert 0 < config["model"]["conf_threshold"] < 1
    assert config["tracking"]["tracker"]
    assert isinstance(config["dataset"]["classes"], list)


def test_load_config_via_package():
    pytest.importorskip("yaml")
    from weedtrack.config import load_config

    config = load_config()
    assert config["model"]["weights"]


def test_fps_meter():
    from weedtrack.pipeline import FpsMeter

    meter = FpsMeter(window=5)
    assert meter.fps == 0.0
    for _ in range(5):
        meter.tick()
        time.sleep(0.01)
    assert meter.fps > 0


def test_fps_meter_fenetre_invalide():
    from weedtrack.pipeline import FpsMeter

    with pytest.raises(ValueError):
        FpsMeter(window=0)


def test_detector_message_clair_sans_ultralytics():
    """Sans ultralytics installé, l'erreur doit guider vers requirements.txt."""
    pytest.importorskip("yaml")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        from weedtrack.detect import Detector

        with pytest.raises(ImportError, match="requirements.txt"):
            Detector("yolo11n.pt")
    else:
        pytest.skip("ultralytics installé : le message d'absence n'est pas testable ici")
