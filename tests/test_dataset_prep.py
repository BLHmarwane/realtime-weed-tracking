"""Tests de la préparation du dataset (conversion VOC → YOLO, split, data.yaml).

Tout est synthétique et en bibliothèque standard : ces tests tournent dans la
CI sans télécharger le vrai dataset.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_dataset import (  # noqa: E402
    VocObject,
    parse_voc_xml,
    prepare_yolo_dataset,
    voc_to_yolo_line,
)

VOC_XML = """<annotation>
\t<filename>{stem}.jpg</filename>
\t<size><width>200</width><height>100</height><depth>3</depth></size>
{objects}
</annotation>
"""

VOC_OBJECT = """\t<object>
\t\t<name>{name}</name>
\t\t<bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
\t</object>
"""


def make_raw_dir(tmp_path: Path, count: int = 10) -> Path:
    """Construit un mini dataset brut synthétique (images factices + XML)."""
    raw = tmp_path / "raw_db"
    (raw / "raw images").mkdir(parents=True)
    (raw / "annotations").mkdir()
    for index in range(count):
        stem = f"img{index:03d}"
        (raw / "raw images" / f"{stem}.jpg").write_bytes(b"fake-jpeg")
        objects = VOC_OBJECT.format(name="weed", xmin=10, ymin=20, xmax=60, ymax=80)
        if index % 2 == 0:
            objects += VOC_OBJECT.format(name="crop", xmin=0, ymin=0, xmax=100, ymax=50)
        (raw / "annotations" / f"{stem}.xml").write_text(
            VOC_XML.format(stem=stem, objects=objects), encoding="utf-8"
        )
    return raw


# ---------- voc_to_yolo_line ----------

def test_conversion_normalisee():
    line = voc_to_yolo_line(VocObject("crop", 0, 0, 100, 50), width=200, height=100)
    assert line == "0 0.250000 0.250000 0.500000 0.500000"


def test_conversion_weed_class_id():
    line = voc_to_yolo_line(VocObject("weed", 50, 25, 150, 75), width=200, height=100)
    assert line is not None and line.startswith("1 ")


def test_boite_hors_image_bornee():
    line = voc_to_yolo_line(VocObject("weed", -50, -10, 400, 300), width=200, height=100)
    assert line == "1 0.500000 0.500000 1.000000 1.000000"


def test_boite_degeneree_rejetee():
    assert voc_to_yolo_line(VocObject("weed", 60, 20, 10, 80), 200, 100) is None


def test_classe_inconnue_rejetee():
    assert voc_to_yolo_line(VocObject("cat", 10, 10, 50, 50), 200, 100) is None


# ---------- parse_voc_xml ----------

def test_parse_xml_valide(tmp_path):
    raw = make_raw_dir(tmp_path, count=1)
    annotation = parse_voc_xml(raw / "annotations" / "img000.xml")
    assert annotation is not None
    assert (annotation.width, annotation.height) == (200, 100)
    assert [o.name for o in annotation.objects] == ["weed", "crop"]


def test_parse_xml_sans_taille(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<annotation><object/></annotation>", encoding="utf-8")
    assert parse_voc_xml(path) is None


# ---------- prepare_yolo_dataset ----------

def test_preparation_complete(tmp_path):
    raw = make_raw_dir(tmp_path, count=10)
    out = tmp_path / "dataset"
    stats = prepare_yolo_dataset(raw, out, val_ratio=0.2, seed=42)

    assert stats["images"] == {"val": 2, "train": 8}
    assert sum(stats["boxes"]["train"].values()) + sum(stats["boxes"]["val"].values()) == 15
    assert stats["dropped_degenerate_boxes"] == 0

    train_images = list((out / "images" / "train").glob("*.jpg"))
    train_labels = list((out / "labels" / "train").glob("*.txt"))
    assert len(train_images) == len(train_labels) == 8

    data_yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data_yaml
    assert "0: crop" in data_yaml and "1: weed" in data_yaml

    # chaque valeur de label est normalisée dans [0, 1]
    sample = train_labels[0].read_text(encoding="utf-8").strip().splitlines()
    for row in sample:
        parts = row.split()
        assert parts[0] in {"0", "1"}
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_split_deterministe(tmp_path):
    raw = make_raw_dir(tmp_path, count=10)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    prepare_yolo_dataset(raw, out_a, val_ratio=0.2, seed=42)
    prepare_yolo_dataset(raw, out_b, val_ratio=0.2, seed=42)
    val_a = sorted(p.name for p in (out_a / "images" / "val").iterdir())
    val_b = sorted(p.name for p in (out_b / "images" / "val").iterdir())
    assert val_a == val_b


def test_relance_ecrase_proprement(tmp_path):
    raw = make_raw_dir(tmp_path, count=4)
    out = tmp_path / "dataset"
    prepare_yolo_dataset(raw, out, val_ratio=0.25, seed=1)
    stats = prepare_yolo_dataset(raw, out, val_ratio=0.25, seed=1)
    assert stats["images"] == {"val": 1, "train": 3}
    assert len(list((out / "images" / "train").iterdir())) == 3
