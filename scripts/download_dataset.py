"""Télécharge, vérifie et prépare le dataset Sudars et al. au format YOLO.

Dataset : « Dataset of annotated food crops and weed images for robotic
computer vision control » (Sudars et al., 2020), Mendeley Data nj4vtk4tt6 v1,
licence CC BY 4.0 — vérifiée avant téléchargement (règle STATE.md).

Étapes (chacune est idempotente : relancer le script ne refait que le manquant) :

1. téléchargement du zip officiel (sauté si déjà présent) ;
2. vérification SHA-256 contre le hash publié par Mendeley (échec = arrêt) ;
3. extraction ;
4. conversion des annotations Pascal VOC (pixels) → labels YOLO (normalisés) ;
5. split train/val déterministe (seed fixe) ;
6. génération de `data/dataset/data.yaml` pour Ultralytics ;
7. statistiques EDA archivées dans `metrics/dataset_stats.json`.

Volontairement 100 % bibliothèque standard : utilisable sans le venv ML.

Usage :
    python scripts/download_dataset.py [--val-ratio 0.2] [--seed 42]
"""

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "Ronin_OPEN_DB.zip"
EXTRACTED_DIR = RAW_DIR / "Ronin_OPEN_DB"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
STATS_PATH = PROJECT_ROOT / "metrics" / "dataset_stats.json"

DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/nj4vtk4tt6/files/"
    "0d3e34be-9f29-4d8d-bd4c-0915ec623847/file_downloaded"
)
EXPECTED_SHA256 = "eca9d015c7d78a94d472dc2832bae3f844c905253c49e6981a23bb7d5b5e0461"

CLASS_TO_ID = {"crop": 0, "weed": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    stem: str
    width: int
    height: int
    objects: list[VocObject]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_zip() -> None:
    if ZIP_PATH.exists():
        print(f"[1/7] Zip déjà présent : {ZIP_PATH}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/7] Téléchargement ({DOWNLOAD_URL}) …")
    urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)  # noqa: S310 — URL constante HTTPS
    print(f"      → {ZIP_PATH}")


def verify_zip() -> None:
    print("[2/7] Vérification SHA-256 …")
    actual = sha256_of(ZIP_PATH)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"SHA-256 inattendu pour {ZIP_PATH} :\n"
            f"  attendu : {EXPECTED_SHA256}\n"
            f"  obtenu  : {actual}\n"
            "Fichier corrompu ou source modifiée — supprimer le zip et relancer."
        )
    print("      OK (conforme au hash officiel Mendeley)")


def ensure_extracted() -> None:
    if EXTRACTED_DIR.is_dir() and any(EXTRACTED_DIR.iterdir()):
        print(f"[3/7] Déjà extrait : {EXTRACTED_DIR}")
        return
    print("[3/7] Extraction …")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(RAW_DIR)
    print(f"      → {EXTRACTED_DIR}")


def parse_voc_xml(path: Path) -> VocAnnotation | None:
    """Parse un XML Pascal VOC ; retourne None si taille absente/invalide."""
    root = ElementTree.parse(path).getroot()
    size = root.find("size")
    if size is None:
        return None
    try:
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None

    objects = []
    for node in root.iter("object"):
        name = (node.findtext("name") or "").strip().lower()
        box = node.find("bndbox")
        if not name or box is None:
            continue
        try:
            objects.append(
                VocObject(
                    name=name,
                    xmin=float(box.findtext("xmin", "0")),
                    ymin=float(box.findtext("ymin", "0")),
                    xmax=float(box.findtext("xmax", "0")),
                    ymax=float(box.findtext("ymax", "0")),
                )
            )
        except ValueError:
            continue
    return VocAnnotation(stem=path.stem, width=width, height=height, objects=objects)


def voc_to_yolo_line(obj: VocObject, width: int, height: int) -> str | None:
    """Convertit une boîte VOC (pixels) en ligne YOLO normalisée, ou None.

    Les coordonnées sont bornées à l'image ; une boîte dégénérée (surface
    nulle après bornage) ou de classe inconnue est rejetée.
    """
    class_id = CLASS_TO_ID.get(obj.name)
    if class_id is None:
        return None
    xmin = min(max(obj.xmin, 0.0), width)
    xmax = min(max(obj.xmax, 0.0), width)
    ymin = min(max(obj.ymin, 0.0), height)
    ymax = min(max(obj.ymax, 0.0), height)
    if xmax - xmin <= 0 or ymax - ymin <= 0:
        return None
    x_center = (xmin + xmax) / 2 / width
    y_center = (ymin + ymax) / 2 / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def build_image_index(images_dir: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES
    }


def prepare_yolo_dataset(
    extracted_dir: Path = EXTRACTED_DIR,
    dataset_dir: Path = DATASET_DIR,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Convertit VOC → YOLO, split train/val déterministe, écrit data.yaml.

    Retourne le dictionnaire de statistiques EDA.
    """
    print("[4/7] Conversion Pascal VOC → YOLO …")
    annotations_dir = extracted_dir / "annotations"
    images_dir = extracted_dir / "raw images"
    image_index = build_image_index(images_dir)

    parsed: list[VocAnnotation] = []
    skipped_xml = 0
    missing_images = 0
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        annotation = parse_voc_xml(xml_path)
        if annotation is None:
            skipped_xml += 1
            continue
        if annotation.stem not in image_index:
            missing_images += 1
            continue
        parsed.append(annotation)

    print(f"[5/7] Split train/val (val_ratio={val_ratio}, seed={seed}) …")
    rng = random.Random(seed)
    shuffled = parsed[:]
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    splits = {"val": shuffled[:val_count], "train": shuffled[val_count:]}

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    box_counts: dict[str, Counter] = {}
    dropped_boxes = 0
    empty_label_images = 0
    boxes_per_image: list[int] = []
    image_sizes = Counter()

    for split_name, annotations in splits.items():
        images_out = dataset_dir / "images" / split_name
        labels_out = dataset_dir / "labels" / split_name
        images_out.mkdir(parents=True)
        labels_out.mkdir(parents=True)
        counts: Counter = Counter()
        for annotation in annotations:
            lines = []
            for obj in annotation.objects:
                line = voc_to_yolo_line(obj, annotation.width, annotation.height)
                if line is None:
                    dropped_boxes += 1
                    continue
                lines.append(line)
                counts[obj.name] += 1
            if not lines:
                empty_label_images += 1
            boxes_per_image.append(len(lines))
            image_sizes[f"{annotation.width}x{annotation.height}"] += 1
            source_image = image_index[annotation.stem]
            shutil.copy2(source_image, images_out / source_image.name)
            (labels_out / f"{annotation.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        box_counts[split_name] = counts

    print("[6/7] Écriture de data.yaml …")
    names_block = "\n".join(
        f"  {class_id}: {name}" for name, class_id in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])
    )
    (dataset_dir / "data.yaml").write_text(
        f"# Généré par scripts/download_dataset.py — ne pas éditer à la main.\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )

    stats = {
        "dataset": "Sudars et al. 2020, Mendeley nj4vtk4tt6 v1 (CC BY 4.0)",
        "seed": seed,
        "val_ratio": val_ratio,
        "images": {name: len(items) for name, items in splits.items()},
        "boxes": {name: dict(counts) for name, counts in box_counts.items()},
        "skipped_xml_without_size": skipped_xml,
        "xml_without_matching_image": missing_images,
        "dropped_degenerate_boxes": dropped_boxes,
        "images_with_zero_valid_boxes": empty_label_images,
        "boxes_per_image": {
            "min": min(boxes_per_image) if boxes_per_image else 0,
            "max": max(boxes_per_image) if boxes_per_image else 0,
            "mean": round(sum(boxes_per_image) / len(boxes_per_image), 2)
            if boxes_per_image
            else 0,
        },
        "image_sizes": dict(image_sizes.most_common()),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_zip()
    verify_zip()
    ensure_extracted()
    stats = prepare_yolo_dataset(val_ratio=args.val_ratio, seed=args.seed)

    print("[7/7] Statistiques EDA …")
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nTerminé : dataset YOLO dans {DATASET_DIR}, stats dans {STATS_PATH}")


if __name__ == "__main__":
    main()
