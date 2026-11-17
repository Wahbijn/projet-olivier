"""API de détection des maladies de l'olivier.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.models.predict import Predicteur

TAILLE_MAX_OCTETS = 10 * 1024 * 1024        # 10 Mo
etat = {"predicteur": None, "erreur": None}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    # Chargement du modèle UNE seule fois, au démarrage de l'API
    try:
        etat["predicteur"] = Predicteur()
    except FileNotFoundError as e:
        etat["erreur"] = f"Modèle introuvable : {e.filename}"
    yield


app = FastAPI(
    title="API Olivier",
    description="Détection de maladies de l'olivier (healthy, aculus, peacock_spot) avec heatmap Grad-CAM.",
    version="1.0.0",
    lifespan=cycle_de_vie,
)

# Autorise l'interface web (React) à appeler l'API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Schémas Pydantic (format des réponses) ----------
class Sante(BaseModel):
    statut: str
    modele_charge: bool
    architecture: str | None = None
    classes: list[str] = []
    seuil_confiance: float | None = None
    erreur: str | None = None


class Prediction(BaseModel):
    classe: str                         # classe finale, ou "incertain" si confiance < seuil
    classe_predite: str                 # classe la plus probable (même si incertain)
    confiance: float
    incertain: bool
    probabilites: dict[str, float]
    seuil_confiance: float
    heatmap_base64: str | None = None   # image PNG (feuille + heatmap) encodée en base64


# ---------- Routes ----------
@app.get("/health", response_model=Sante)
def health():
    p = etat["predicteur"]
    if p is None:
        return Sante(statut="degrade", modele_charge=False, erreur=etat["erreur"])
    return Sante(statut="ok", modele_charge=True, architecture=p.architecture,
                 classes=p.classes, seuil_confiance=p.seuil)


@app.post("/predict", response_model=Prediction)
async def predict(fichier: UploadFile = File(...), heatmap: bool = True):
    p = etat["predicteur"]
    if p is None:
        raise HTTPException(status_code=503, detail=etat["erreur"] or "Modèle non chargé.")

    contenu = await fichier.read()
    if len(contenu) > TAILLE_MAX_OCTETS:
        raise HTTPException(status_code=413, detail="Image trop lourde (10 Mo maximum).")
    try:
        img = Image.open(io.BytesIO(contenu))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide.")

    # Calcul lourd : exécuté hors de la boucle principale pour ne pas bloquer l'API
    return await run_in_threadpool(p.predire, img, heatmap)
