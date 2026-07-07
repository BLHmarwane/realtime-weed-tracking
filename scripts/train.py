"""Fine-tune le modèle YOLO sur le dataset préparé par download_dataset.py.

Prévu au milestone M2. Lit tout depuis configs/config.yaml : pas
d'hyperparamètre en dur ici, pour que chaque run soit reproductible.

Usage cible :
    python scripts/train.py [--config configs/config.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weedtrack.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="chemin du YAML de config")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    del config  # utilisé à M2

    raise SystemExit(
        "M2 non commencé : nécessite le dataset (M1). Implémentation prévue : "
        "YOLO(weights).train(data=..., epochs=..., imgsz=..., device=...) "
        "puis copie des poids best.pt vers models/."
    )


if __name__ == "__main__":
    main()
