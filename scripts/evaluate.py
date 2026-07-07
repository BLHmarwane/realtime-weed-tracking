"""Évalue les poids fine-tunés et archive les métriques officielles.

Règle du projet (STATE.md) : tout chiffre publié dans le README sort d'ici et
est archivé dans metrics/ — jamais écrit à la main. Le script produit :

- mAP@50 et mAP@50-95 globaux et par classe (val split) ;
- un benchmark FPS d'inférence honnête (weedtrack.pipeline.FpsMeter) sur des
  images du val, avec device et taille d'image documentés.

Usage :
    python scripts/evaluate.py [--weights models/best.pt] [--device mps]
                               [--out metrics/baseline.json]
"""

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weedtrack.config import load_config  # noqa: E402
from weedtrack.pipeline import FpsMeter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_metrics_payload(
    *,
    map50: float,
    map50_95: float,
    per_class: dict[str, dict[str, float]],
    fps: float | None,
    fps_images: int,
    device: str,
    imgsz: int,
    weights: str,
    ultralytics_version: str,
    machine: str,
) -> dict:
    """Assemble le JSON de métriques (fonction pure, testée unitairement)."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "weights": weights,
        "dataset": "Sudars et al. 2020 (Mendeley nj4vtk4tt6, CC BY 4.0), split val",
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": {
            name: {metric: round(value, 4) for metric, value in values.items()}
            for name, values in per_class.items()
        },
        "inference": {
            "fps": round(fps, 2) if fps is not None else None,
            "benchmark_images": fps_images,
            "device": device,
            "imgsz": imgsz,
            "machine": machine,
        },
        "ultralytics_version": ultralytics_version,
    }


def benchmark_fps(model, images: list[Path], device: str, imgsz: int, warmup: int = 5) -> float | None:
    """FPS moyen de prédiction image par image (mesure, pas estimation)."""
    if not images:
        return None
    for path in images[:warmup]:
        model.predict(str(path), device=device, imgsz=imgsz, verbose=False)
    meter = FpsMeter(window=len(images))
    meter.tick()
    for path in images:
        model.predict(str(path), device=device, imgsz=imgsz, verbose=False)
        meter.tick()
    return meter.fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "best.pt"))
    parser.add_argument("--device", default=None, help="défaut : config train.device")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "metrics" / "baseline.json"))
    parser.add_argument("--bench-images", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    device = args.device or config["train"]["device"]
    imgsz = config["model"]["imgsz"]
    data_yaml = PROJECT_ROOT / config["dataset"]["data_yaml"]

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(args.weights)

    print(f"Validation ({device}, imgsz={imgsz}) …")
    results = model.val(
        data=str(data_yaml),
        device=device,
        imgsz=imgsz,
        verbose=False,
        # confine les sorties ultralytics au dossier du projet (sinon elles
        # atterrissent dans le cwd de l'appelant)
        project=str(PROJECT_ROOT / "runs"),
        name="val",
        exist_ok=True,
    )

    names = results.names
    per_class: dict[str, dict[str, float]] = {}
    for position, class_index in enumerate(results.box.ap_class_index):
        per_class[names[int(class_index)]] = {
            "ap50": float(results.box.ap50[position]),
            "ap50_95": float(results.box.ap[position]),
        }

    val_images = sorted((data_yaml.parent / "images" / "val").glob("*"))[: args.bench_images]
    print(f"Benchmark FPS sur {len(val_images)} images val …")
    fps = benchmark_fps(model, val_images, device=device, imgsz=imgsz)

    payload = build_metrics_payload(
        map50=float(results.box.map50),
        map50_95=float(results.box.map),
        per_class=per_class,
        fps=fps,
        fps_images=len(val_images),
        device=str(device),
        imgsz=imgsz,
        weights=Path(args.weights).name,
        ultralytics_version=ultralytics.__version__,
        machine=f"{platform.system()} {platform.machine()}",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nMétriques archivées : {out_path}")


if __name__ == "__main__":
    main()
