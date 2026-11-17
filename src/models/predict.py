"""Prédiction sur une photo de feuille d'olivier, avec heatmap Grad-CAM.

Le prétraitement est identique à celui du notebook (section 5) :
padding carré avec la couleur du fond, redimensionnement en 224x224, normalisation ImageNet.
"""
import base64
import io
import threading
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision import models

RACINE = Path(__file__).resolve().parents[2]      # dossier projet-olivier


def charger_config(chemin=RACINE / "configs" / "config.yaml"):
    with open(chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparer_image(img: Image.Image, taille: int = 224) -> np.ndarray:
    """Rend l'image carrée (padding couleur du fond) puis la met en taille x taille."""
    img = img.convert("RGB")
    a = np.asarray(img)
    # couleur du fond = médiane des 5 premières et 5 dernières lignes de pixels
    bord = np.concatenate([a[:5].reshape(-1, 3), a[-5:].reshape(-1, 3)])
    couleur = tuple(int(v) for v in np.median(bord, axis=0))
    img = ImageOps.pad(img, (taille, taille), color=couleur, method=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def image_en_base64(tableau: np.ndarray) -> str:
    """Tableau d'image (H, W, 3) -> texte PNG encodé en base64."""
    tampon = io.BytesIO()
    Image.fromarray(tableau).save(tampon, format="PNG")
    return base64.b64encode(tampon.getvalue()).decode("ascii")


class Predicteur:
    """Charge le modèle une seule fois, puis prédit image par image."""

    def __init__(self, config: dict | None = None):
        self.config = config or charger_config()
        infos = self.config["modele_deploye"]
        ckpt = torch.load(RACINE / infos["pytorch"], map_location="cpu", weights_only=False)

        self.architecture = ckpt["architecture"]
        self.classes = ckpt["classes"]
        self.taille = ckpt["taille_entree"]
        self.moyenne = torch.tensor(ckpt["moyenne"]).view(3, 1, 1)
        self.ecart = torch.tensor(ckpt["ecart"]).view(3, 1, 1)
        self.seuil = float(infos.get("seuil_confiance", ckpt["seuil_confiance"]))

        modele = models.resnet50()
        modele.fc = nn.Linear(modele.fc.in_features, len(self.classes))
        modele.load_state_dict(ckpt["state_dict"])
        modele.eval()
        for p in modele.parameters():           # pas besoin de gradients sur les poids
            p.requires_grad_(False)
        self.modele = modele

        # "Micro" Grad-CAM sur la dernière couche convolutive
        self._activations, self._gradients = None, None
        modele.layer4[-1].register_forward_hook(self._garder)
        self._verrou = threading.Lock()          # une prédiction à la fois (les hooks gardent un état)

    def _garder(self, module, entree, sortie):
        self._activations = sortie.detach()
        if sortie.requires_grad:
            sortie.register_hook(lambda g: setattr(self, "_gradients", g.detach()))

    def _en_tenseur(self, tableau: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(tableau.copy()).permute(2, 0, 1).float() / 255.0
        return ((x - self.moyenne) / self.ecart).unsqueeze(0)

    def predire(self, img: Image.Image, avec_heatmap: bool = True) -> dict:
        tableau = preparer_image(img, self.taille)
        x = self._en_tenseur(tableau)

        with self._verrou:
            if avec_heatmap:
                x.requires_grad_(True)           # nécessaire pour calculer Grad-CAM
                sortie = self.modele(x)
            else:
                with torch.no_grad():
                    sortie = self.modele(x)

            probas = torch.softmax(sortie.detach(), dim=1)[0]
            idx = int(probas.argmax())
            confiance = float(probas[idx])
            incertain = confiance < self.seuil

            resultat = {
                "classe": "incertain" if incertain else self.classes[idx],
                "classe_predite": self.classes[idx],
                "confiance": round(confiance, 4),
                "incertain": incertain,
                "probabilites": {c: round(float(p), 4) for c, p in zip(self.classes, probas)},
                "seuil_confiance": self.seuil,
                "heatmap_base64": None,
            }

            if avec_heatmap:
                sortie[0, idx].backward()
                poids = self._gradients.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((poids * self._activations).sum(dim=1)).squeeze()
                cam = cam / (cam.max() + 1e-8)
                cam = F.interpolate(cam[None, None], size=(self.taille, self.taille),
                                    mode="bilinear", align_corners=False).squeeze().numpy()
                couleurs = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
                superpose = (0.55 * tableau + 0.45 * couleurs).astype(np.uint8)
                resultat["heatmap_base64"] = image_en_base64(superpose)

        return resultat
