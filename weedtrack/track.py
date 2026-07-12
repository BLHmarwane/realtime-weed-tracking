"""Multi-object tracking across video frames.

ByteTrack associates short-term IDs across adjacent frames.
It does not prove long-term identity stability or guarantee treatment deduplication.
"""

from collections.abc import Iterator


def iter_tracked_results(
    source: str,
    weights: str,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
) -> Iterator:
    """Itère sur les résultats détection+tracking d'une vidéo, frame par frame.

    `source` : chemin vidéo, dossier d'images ou index de webcam.
    Chaque résultat Ultralytics porte `boxes.id` (l'identifiant de piste).

    Les IDs représentent des associations à court terme et peuvent se
    fragmenter ; le débit mesuré est archivé par les scripts d'évaluation.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics n'est pas installé. Créer le venv du projet puis "
            "`pip install -r requirements.txt` (voir README.md)."
        ) from exc

    model = YOLO(weights)
    yield from model.track(
        source,
        tracker=tracker,
        conf=conf,
        imgsz=imgsz,
        device=device,
        stream=True,   # générateur : la vidéo n'est jamais chargée entière en RAM
        verbose=False,
    )
