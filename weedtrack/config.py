"""Chargement de la configuration YAML unique du projet.

Toute la chaîne (train, eval, tracking, démo) lit `configs/config.yaml` via ce
module : un seul endroit à modifier pour changer de modèle, de seuils ou de
dataset.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Lit le YAML de configuration et le retourne en dict.

    L'import de yaml est paresseux pour que `import weedtrack` fonctionne sans
    dépendances installées.
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration invalide ou vide : {path}")
    return config
