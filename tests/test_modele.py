"""Vérifie le modèle exporté (remplace l'ancien script tests/verifier_modele.py).

Lancement : pytest tests/test_modele.py -v -s
Ignoré automatiquement si models/model_V1.pt / .onnx sont absents (ex. CI).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.architectures import creer_modele
from src.utils import charger_config

RACINE = Path(__file__).resolve().parents[1]
PT = RACINE / "models" / "model_V1.pt"
ONNX = RACINE / "models" / "model_V1.onnx"

pytestmark = pytest.mark.skipif(not PT.exists(), reason="models/model_V1.pt absent (dvc pull)")


@pytest.fixture(scope="module")
def ckpt():
    return torch.load(PT, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def modele(ckpt):
    m = creer_modele(ckpt["architecture"], len(ckpt["classes"]), pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def test_metadonnees(ckpt):
    for cle in ["state_dict", "architecture", "classes", "taille_entree", "moyenne", "ecart",
                "seuil_confiance", "metriques"]:
        assert cle in ckpt, f"clé manquante : {cle}"
    assert ckpt["classes"] == charger_config()["classes"]
    print("\nArchitecture :", ckpt["architecture"], "| seuil :", ckpt["seuil_confiance"],
          "| métriques :", ckpt["metriques"])


@pytest.mark.skipif(not ONNX.exists(), reason="models/model_V1.onnx absent")
def test_onnx_identique_au_pytorch(modele):
    import onnxruntime as ort
    x = np.random.rand(2, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        sortie_pt = modele(torch.from_numpy(x)).numpy()
    sortie_onnx = ort.InferenceSession(str(ONNX)).run(None, {"image": x})[0]
    ecart = np.abs(sortie_pt - sortie_onnx).max()
    print("\nÉcart max PyTorch vs ONNX :", ecart)
    assert ecart < 1e-3
