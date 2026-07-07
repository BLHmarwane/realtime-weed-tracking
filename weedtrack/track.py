"""Tracking multi-objets : suivi des plantes d'une frame à l'autre.

Choix par défaut (voir STATE.md « Questions ouvertes ») : le tracking intégré
d'Ultralytics avec ByteTrack (`model.track`), qui associe un identifiant stable
à chaque objet au fil de la vidéo. C'est ce qui distingue « je détecte une
adventice » de « je sais que c'est LA MÊME adventice que sur la frame
précédente » — indispensable pour ne pas traiter deux fois la même plante.
"""

from collections.abc import Iterator


def iter_tracked_results(
    source: str,
    weights: str,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int = 640,
) -> Iterator:
    """Itère sur les résultats détection+tracking d'une vidéo, frame par frame.

    `source` : chemin vidéo, dossier d'images ou index de webcam.
    Chaque résultat Ultralytics porte `boxes.id` (l'identifiant de piste).

    TODO(M3) : mesurer la stabilité des IDs (fragmentation de pistes) et le
    FPS réel via `weedtrack.pipeline.FpsMeter`, puis figer les seuils.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics n'est pas installé. Créer le venv du projet puis "
            "`pip install -r requirements.txt` (voir TUTORIAL.md §7)."
        ) from exc

    model = YOLO(weights)
    yield from model.track(
        source,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        stream=True,   # générateur : la vidéo n'est jamais chargée entière en RAM
        verbose=False,
    )
