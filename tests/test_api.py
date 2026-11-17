"""Tests automatiques de l'API (lancement : pytest)."""
import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from api.main import app

MODELE = Path(__file__).resolve().parents[1] / "models" / "model_V1.pt"
besoin_modele = pytest.mark.skipif(not MODELE.exists(),
                                   reason="models/model_V1.pt absent (non versionné dans Git)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # "with" déclenche le chargement du modèle
        yield c


def fausse_feuille() -> bytes:
    """Image de test : une ellipse verte sur fond gris clair, au format JPEG."""
    img = Image.new("RGB", (800, 600), (210, 210, 205))
    ImageDraw.Draw(img).ellipse((150, 250, 650, 350), fill=(70, 110, 60))
    tampon = io.BytesIO()
    img.save(tampon, format="JPEG")
    return tampon.getvalue()


def test_health_repond(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["statut"] in ("ok", "degrade")


@besoin_modele
def test_health_modele_charge(client):
    d = client.get("/health").json()
    assert d["modele_charge"] is True
    assert d["classes"] == ["healthy", "aculus", "peacock_spot"]


@besoin_modele
def test_predict_avec_heatmap(client):
    r = client.post("/predict", files={"fichier": ("feuille.jpg", fausse_feuille(), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert d["classe_predite"] in ["healthy", "aculus", "peacock_spot"]
    assert d["classe"] in ["healthy", "aculus", "peacock_spot", "incertain"]
    assert 0.0 <= d["confiance"] <= 1.0
    assert abs(sum(d["probabilites"].values()) - 1.0) < 1e-3
    heatmap = Image.open(io.BytesIO(base64.b64decode(d["heatmap_base64"])))
    assert heatmap.size == (224, 224)


@besoin_modele
def test_predict_sans_heatmap(client):
    r = client.post("/predict", params={"heatmap": False},
                    files={"fichier": ("feuille.jpg", fausse_feuille(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["heatmap_base64"] is None


@besoin_modele
def test_fichier_invalide(client):
    r = client.post("/predict", files={"fichier": ("texte.txt", b"ceci n'est pas une image", "text/plain")})
    assert r.status_code == 400
