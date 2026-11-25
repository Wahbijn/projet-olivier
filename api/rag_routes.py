"""Routes de l'assistant agronomique RAG : POST /ask et GET /rag/health.

Le RAG est optionnel : si l'index n'est pas construit, /predict continue de fonctionner
et /ask répond 503 avec la marche à suivre.
"""
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.rag.llm import ErreurLLM

router = APIRouter(tags=["assistant RAG"])
etat_rag = {"assistant": None, "erreur": None}


def charger_rag() -> None:
    """Appelé au démarrage de l'API. Ne lève jamais d'exception : l'erreur est gardée dans etat_rag."""
    try:
        from src.rag.pipeline import charger_assistant      # import ici : évite de charger numpy & co si inutile
        etat_rag["assistant"] = charger_assistant()
        etat_rag["erreur"] = None
    except FileNotFoundError:
        etat_rag["erreur"] = "Index RAG absent : lancer `python -m src.rag.construire_index`."
    except Exception as e:                                  # dépendance manquante, config invalide...
        etat_rag["erreur"] = f"RAG indisponible : {e}"


# ---------- Schémas ----------
class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    classe: Literal["healthy", "aculus", "peacock_spot", "incertain"] | None = None   # diagnostic de la photo
    confiance: float | None = Field(None, ge=0.0, le=1.0)
    langue: Literal["fr", "ar"] = "fr"


class SourceCitee(BaseModel):
    numero: int
    titre: str
    url: str | None = None
    extrait: str
    score: float


class ReponseRAG(BaseModel):
    reponse: str
    sources: list[SourceCitee] = []
    refuse: bool = False        # True : rien de pertinent trouvé, le LLM n'a pas été appelé


class SanteRAG(BaseModel):
    pret: bool
    passages: int | None = None
    embeddings: str | None = None
    erreur: str | None = None


# ---------- Routes ----------
@router.get("/rag/health", response_model=SanteRAG)
def rag_health():
    a = etat_rag["assistant"]
    if a is None:
        return SanteRAG(pret=False, erreur=etat_rag["erreur"])
    return SanteRAG(pret=True, passages=len(a.index), embeddings=a.index.embedder_nom)


@router.post("/ask", response_model=ReponseRAG)
async def ask(q: Question):
    a = etat_rag["assistant"]
    if a is None:
        raise HTTPException(status_code=503, detail=etat_rag["erreur"] or "Assistant non chargé.")
    try:
        # Embeddings + appel LLM : bloquants, donc hors de la boucle principale
        rep = await run_in_threadpool(a.repondre, q.question.strip(), q.classe, q.confiance, q.langue)
    except ErreurLLM as e:
        raise HTTPException(status_code=502, detail=f"Le modèle de langage n'a pas répondu : {e}")
    except Exception as e:                                  # bug inattendu : on garde la trace et on l'explique
        logging.getLogger("uvicorn.error").exception("Erreur dans /ask")
        raise HTTPException(status_code=500, detail=f"Erreur interne ({type(e).__name__}) : {e}")
    return ReponseRAG(
        reponse=rep.texte,
        sources=[SourceCitee(**vars(s)) for s in rep.sources],
        refuse=rep.refuse,
    )
