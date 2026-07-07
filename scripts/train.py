"""Fine-tune le modèle YOLO sur le dataset préparé par download_dataset.py.

Lit tout depuis configs/config.yaml (surchargeable en CLI pour les probes) :
pas d'hyperparamètre caché, chaque run est reproductible et journalisé dans
runs/ (hors Git). À la fin, les meilleurs poids sont copiés vers
models/best.pt — jamais commités.

Usage :
    python scripts/train.py                      # run complet selon la config
    python scripts/train.py --epochs 1 --fraction 0.25 --device mps --name probe
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weedtrack.config import load_config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="chemin du YAML de config")
    parser.add_argument("--epochs", type=int, default=None, help="surcharge config")
    parser.add_argument("--batch", type=int, default=None, help="surcharge config")
    parser.add_argument("--device", default=None, help="surcharge config (cpu, mps, 0…)")
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="fraction du train utilisée (probe de vitesse, ex. 0.25)",
    )
    parser.add_argument("--name", default="baseline", help="nom du run dans runs/")
    parser.add_argument(
        "--patience", type=int, default=15,
        help="arrêt anticipé si val sans progrès pendant N époques",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    data_yaml = PROJECT_ROOT / config["dataset"]["data_yaml"]
    if not data_yaml.exists():
        raise SystemExit(
            f"{data_yaml} introuvable — lancer d'abord scripts/download_dataset.py (M1)."
        )

    from ultralytics import YOLO

    model = YOLO(config["model"]["weights"])
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs or config["train"]["epochs"],
        imgsz=config["model"]["imgsz"],
        batch=args.batch or config["train"]["batch"],
        device=args.device or config["train"]["device"],
        fraction=args.fraction,
        patience=args.patience,
        project=str(PROJECT_ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = PROJECT_ROOT / "models" / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"\nMeilleurs poids : {best}\nCopiés vers    : {dest}")
    else:
        print(f"\nAttention : {best} absent (run interrompu ?) — rien copié.")


if __name__ == "__main__":
    main()
