"""Génère une vidéo de test à partir des images val du dataset (CC BY 4.0).

Pourquoi générer plutôt que télécharger une vidéo stock ? Décision documentée
dans STATE.md : les vidéos agricoles libres (Pexels/Pixabay) montrent des
cultures adultes filmées en drone ou à hauteur d'homme — hors distribution
pour un modèle entraîné sur des plantules vues de dessus. À l'inverse, un
balayage simulé des images val reproduit le point de vue d'une caméra de
robot de désherbage qui avance au-dessus du rang : les plantes entrent et
sortent du champ de vision de façon continue, ce qui est exactement le cas
d'usage du tracking. Même source, même licence (CC BY 4.0), zéro ambiguïté.

Usage :
    python scripts/make_test_video.py [--num-images 6] [--seconds 4]
                                      [--fps 25] [--out data/test_video.mp4]
"""

import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_IMAGES_DIR = PROJECT_ROOT / "data" / "dataset" / "images" / "val"
DEFAULT_OUT = PROJECT_ROOT / "data" / "test_video.mp4"

WINDOW = 640          # taille de la fenêtre de balayage (= imgsz du modèle)
MIN_EXTRA_WIDTH = 100  # il faut de la marge horizontale pour que le pan existe


def pan_positions(image_width: int, image_height: int, window: int, steps: int) -> list[tuple[int, int]]:
    """Trajectoire de la fenêtre : pan horizontal gauche → droite, y centré.

    Fonction pure (testée unitairement). Retourne `steps` positions (x, y)
    entières ; la première colle au bord gauche, la dernière au bord droit.
    """
    if steps < 2:
        raise ValueError("steps doit être >= 2")
    if image_width < window or image_height < window:
        raise ValueError("image plus petite que la fenêtre")
    x_max = image_width - window
    y = (image_height - window) // 2
    return [(round(index * x_max / (steps - 1)), y) for index in range(steps)]


def eligible_images(images_dir: Path) -> list[Path]:
    """Images val assez grandes pour un balayage (import cv2 paresseux)."""
    import cv2

    eligible = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if width >= WINDOW + MIN_EXTRA_WIDTH and height >= WINDOW:
            eligible.append(path)
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=4.0, help="durée du pan par image")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    import cv2

    candidates = eligible_images(VAL_IMAGES_DIR)
    if len(candidates) < args.num_images:
        raise SystemExit(
            f"Seulement {len(candidates)} images val éligibles "
            f"(>= {WINDOW + MIN_EXTRA_WIDTH}x{WINDOW}) — lancer download_dataset.py d'abord ?"
        )
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.num_images)

    steps = max(2, int(args.seconds * args.fps))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WINDOW, WINDOW)
    )

    total_frames = 0
    for path in selected:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        for x, y in pan_positions(width, height, WINDOW, steps):
            writer.write(image[y : y + WINDOW, x : x + WINDOW])
            total_frames += 1
    writer.release()

    duration = total_frames / args.fps
    print(f"Vidéo générée : {out_path}")
    print(f"  {args.num_images} images val (seed {args.seed}), {total_frames} frames, "
          f"{duration:.1f}s à {args.fps} fps, fenêtre {WINDOW}x{WINDOW}")
    print("  Source : dataset Sudars et al. 2020, CC BY 4.0 (hors Git).")


if __name__ == "__main__":
    main()
