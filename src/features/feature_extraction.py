"""Préparation des images (notebook, section 5 / cellule 8).

Ce module est la SEULE définition du prétraitement : l'entraînement (src/models/train.py)
et l'API (src/models/predict.py) l'importent tous les deux, ce qui garantit que le modèle
voit en production exactement les mêmes images qu'à l'entraînement.

Lancement (étape « preparation » du pipeline DVC) :
    python -m src.features.feature_extraction
Produit data/processed/images_224.npy : toutes les images de splits.csv, dans le même ordre.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2 as T

from src.utils import RACINE, charger_config, charger_params

MOYENNE = [0.485, 0.456, 0.406]       # normalisation ImageNet
ECART = [0.229, 0.224, 0.225]


def preparer_image(img: Image.Image, taille: int = 224) -> np.ndarray:
    """Rend l'image carrée (padding couleur du fond) puis la met en taille x taille."""
    img = img.convert("RGB")
    a = np.asarray(img)
    # couleur du fond = médiane des 5 premières et 5 dernières lignes de pixels
    bord = np.concatenate([a[:5].reshape(-1, 3), a[-5:].reshape(-1, 3)])
    couleur = tuple(int(v) for v in np.median(bord, axis=0))
    img = ImageOps.pad(img, (taille, taille), color=couleur, method=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def transformations(taille: int = 224, color_jitter: dict | None = None):
    """Transformations d'entraînement (augmentation) et d'évaluation, comme dans le notebook."""
    params = charger_params()["augmentation"]
    jitter = color_jitter or params["color_jitter_standard"]
    train = T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(taille, scale=tuple(params["random_resized_crop_scale"]), antialias=True),
        *([T.RandomHorizontalFlip()] if params["flip_horizontal"] else []),
        *([T.RandomVerticalFlip()] if params["flip_vertical"] else []),
        T.ColorJitter(**jitter),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MOYENNE, ECART),
    ])
    evaluation = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MOYENNE, ECART),
    ])
    return train, evaluation


def chemin_images_preparees():
    return RACINE / "data" / "processed" / "images_224.npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import pandas as pd          # import local : l'API n'a pas besoin de pandas

    config = charger_config()
    taille = config["pretraitement"]["taille_image"]
    raw = RACINE / config["donnees"]["raw"]
    splits = pd.read_csv(RACINE / config["donnees"]["splits"])

    def charger(chemin_relatif):
        with Image.open(raw / chemin_relatif) as img:
            return preparer_image(img, taille)

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        X = np.stack(list(ex.map(charger, splits["chemin"])))

    sortie = chemin_images_preparees()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.save(sortie, X)
    print(f"{X.shape[0]} images préparées {X.shape[1:]} -> {sortie.relative_to(RACINE)} "
          f"({X.nbytes / 1e6:.0f} Mo)")


if __name__ == "__main__":
    main()
