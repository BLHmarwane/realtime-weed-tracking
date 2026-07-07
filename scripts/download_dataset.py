"""Télécharge et prépare le dataset au format YOLO (images/ + labels/ + data.yaml).

Prévu au milestone M1. La règle (STATE.md) : la licence du dataset est vérifiée
et notée dans STATE.md AVANT tout téléchargement.

Usage cible :
    python scripts/download_dataset.py --dest data/dataset
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/dataset", help="dossier de destination")
    parser.parse_args()
    raise SystemExit(
        "M1 non commencé : choisir le dataset (licence d'abord — voir STATE.md "
        "« Questions ouvertes ») puis implémenter ce script."
    )


if __name__ == "__main__":
    main()
