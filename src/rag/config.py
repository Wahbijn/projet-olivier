"""Lecture de configs/rag.yaml."""
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]      # dossier projet-olivier
CHEMIN_CONFIG = RACINE / "configs" / "rag.yaml"


def charger_config_rag(chemin: Path = CHEMIN_CONFIG) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)["rag"]


def chemin_projet(relatif: str) -> Path:
    """Transforme un chemin relatif de la config en chemin absolu depuis la racine du projet."""
    p = Path(relatif)
    return p if p.is_absolute() else RACINE / p
