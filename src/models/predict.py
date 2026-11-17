"""Prédiction sur une photo de feuille d'olivier, avec heatmap Grad-CAM.

Le prétraitement est importé de src/features/feature_extraction.py : c'est exactement
le même code qu'à l'entraînement (padding carré couleur du fond, 224x224, normalisation ImageNet).
"""
import base64
import io
import threading

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.features.feature_extraction import preparer_image  # noqa: F401  (réexporté pour l'API)
from src.models.architectures import couche_gradcam, creer_modele
from src.utils import RACINE, charger_config


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

        modele = creer_modele(self.architecture, len(self.classes), pretrained=False)
        modele.load_state_dict(ckpt["state_dict"])
        modele.eval()
        for p in modele.parameters():           # pas besoin de gradients sur les poids
            p.requires_grad_(False)
        self.modele = modele

        # "Micro" Grad-CAM sur la dernière couche convolutive
        self._activations, self._gradients = None, None
        couche_gradcam(modele, self.architecture).register_forward_hook(self._garder)
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
