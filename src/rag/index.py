"""Index vectoriel minimaliste : une matrice numpy + la liste des passages (JSON).

Pour quelques milliers de passages, un produit matriciel suffit largement : pas besoin de base
vectorielle. Si le corpus dépasse ~50 000 passages, passer à FAISS ou Chroma.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .decoupage import Passage

FICHIER_VECTEURS = "vecteurs.npy"
FICHIER_PASSAGES = "passages.json"


@dataclass
class Resultat:
    passage: Passage
    score: float


class Index:
    def __init__(self, vecteurs: np.ndarray, passages: list[Passage], embedder_nom: str):
        if len(vecteurs) != len(passages):
            raise ValueError("Nombre de vecteurs et de passages différent.")
        self.vecteurs = vecteurs
        self.passages = passages
        self.embedder_nom = embedder_nom

    def __len__(self):
        return len(self.passages)

    def chercher(self, vecteur_requete: np.ndarray, k: int = 5) -> list[Resultat]:
        if not len(self):
            return []
        scores = self.vecteurs @ vecteur_requete
        k = min(k, len(scores))
        meilleurs = np.argpartition(-scores, k - 1)[:k]
        meilleurs = meilleurs[np.argsort(-scores[meilleurs])]
        return [Resultat(self.passages[i], float(scores[i])) for i in meilleurs]

    # ---------- sauvegarde / chargement ----------
    def sauvegarder(self, dossier: Path) -> None:
        dossier.mkdir(parents=True, exist_ok=True)
        np.save(dossier / FICHIER_VECTEURS, self.vecteurs)
        with open(dossier / FICHIER_PASSAGES, "w", encoding="utf-8") as f:
            json.dump({"embedder": self.embedder_nom, "passages": [asdict(p) for p in self.passages]},
                      f, ensure_ascii=False)

    @classmethod
    def charger(cls, dossier: Path) -> "Index":
        chemin_v, chemin_p = dossier / FICHIER_VECTEURS, dossier / FICHIER_PASSAGES
        if not chemin_v.exists() or not chemin_p.exists():
            raise FileNotFoundError(f"Index RAG introuvable dans {dossier} (lancer : python -m src.rag.construire_index)")
        with open(chemin_p, encoding="utf-8") as f:
            contenu = json.load(f)
        passages = [Passage(**p) for p in contenu["passages"]]
        return cls(np.load(chemin_v), passages, contenu["embedder"])

    @classmethod
    def construire(cls, passages: list[Passage], embedder) -> "Index":
        vecteurs = embedder.encoder_passages([p.texte for p in passages])
        return cls(vecteurs, passages, embedder.nom)
