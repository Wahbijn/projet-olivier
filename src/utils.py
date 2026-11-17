"""Outils communs : chemins du projet, lecture des configurations, graine aléatoire."""
import hashlib
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

RACINE = Path(__file__).resolve().parents[1]          # dossier projet-olivier


def charger_yaml(nom: str) -> dict:
    """Lit configs/<nom> (ex. "config.yaml", "param.yaml")."""
    with open(RACINE / "configs" / nom, encoding="utf-8") as f:
        return yaml.safe_load(f)


def charger_config() -> dict:
    return charger_yaml("config.yaml")


def charger_params() -> dict:
    return charger_yaml("param.yaml")


def fixer_graine(graine: int):
    """Rend l'entraînement reproductible (autant que le GPU le permet)."""
    import torch
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def empreinte_fichier(chemin: Path) -> str:
    """MD5 d'un fichier : sert de « version » du découpage dans MLflow."""
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RACINE,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "inconnu"
