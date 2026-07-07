"""Évalue le modèle (mAP sur le split val) et archive les métriques.

Prévu au milestone M2/M3. Règle du projet (STATE.md) : toute métrique publiée
dans le README doit sortir de ce script et être archivée dans metrics/
(ex. metrics/baseline.json) — jamais un chiffre écrit à la main.

Usage cible :
    python scripts/evaluate.py --weights models/best.pt --out metrics/baseline.json
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--out", default="metrics/baseline.json")
    parser.parse_args()
    raise SystemExit(
        "M2 non commencé : implémenter model.val() -> mAP@50, mAP@50-95 par "
        "classe, export JSON horodaté + benchmark FPS (weedtrack.pipeline.FpsMeter)."
    )


if __name__ == "__main__":
    main()
