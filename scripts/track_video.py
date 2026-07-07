"""Exécute le pipeline détection + tracking sur une vidéo et archive les stats.

Produit :
- une vidéo annotée (boîtes colorées par classe + IDs de piste) dans runs/ ;
- les statistiques officielles dans metrics/tracking.json (règle du projet :
  tout chiffre publié sort d'un script et vit dans metrics/).

Usage :
    python scripts/track_video.py [--source data/test_video.mp4]
                                  [--out runs/tracking/annotated.mp4]
                                  [--device mps] [--metrics metrics/tracking.json]
"""

import argparse
import datetime
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weedtrack.config import load_config  # noqa: E402
from weedtrack.pipeline import run_on_video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_tracking_payload(
    *,
    summary: dict,
    source: str,
    tracker: str,
    device: str,
    imgsz: int,
    conf: float,
    weights: str,
) -> dict:
    """Assemble le JSON de stats tracking (fonction pure, testée)."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source": source,
        "weights": weights,
        "tracker": tracker,
        "conf_threshold": conf,
        "imgsz": imgsz,
        "device": device,
        "machine": f"{platform.system()} {platform.machine()}",
        "results": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(PROJECT_ROOT / "data" / "test_video.mp4"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "tracking" / "annotated.mp4"))
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "best.pt"))
    parser.add_argument("--device", default=None, help="défaut : config train.device")
    parser.add_argument("--metrics", default=str(PROJECT_ROOT / "metrics" / "tracking.json"))
    args = parser.parse_args()

    if not Path(args.source).exists():
        raise SystemExit(f"{args.source} introuvable — lancer scripts/make_test_video.py d'abord.")
    if not Path(args.weights).exists():
        raise SystemExit(f"{args.weights} introuvable — lancer scripts/train.py d'abord (M2).")

    config = load_config()
    device = args.device or config["train"]["device"]
    tracker = config["tracking"]["tracker"]
    conf = config["model"]["conf_threshold"]
    imgsz = config["model"]["imgsz"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Tracking ({tracker}, {device}, conf={conf}) sur {args.source} …")
    summary = run_on_video(
        source=args.source,
        output_path=str(out_path),
        weights=args.weights,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

    payload = build_tracking_payload(
        summary=summary,
        source=Path(args.source).name,
        tracker=tracker,
        device=str(device),
        imgsz=imgsz,
        conf=conf,
        weights=Path(args.weights).name,
    )
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nVidéo annotée : {out_path}\nStats archivées : {metrics_path}")


if __name__ == "__main__":
    main()
