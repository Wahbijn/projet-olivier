"""Embeddings : transforment un texte en vecteur normalisé (le produit scalaire = similarité cosinus)."""
import hashlib
import re
import unicodedata

import numpy as np


class Embedder:
    nom: str = "base"

    def encoder_passages(self, textes: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encoder_requete(self, texte: str) -> np.ndarray:
        raise NotImplementedError


class EmbedderSentenceTransformers(Embedder):
    """Modèle multilingue local (FR / AR / EN). Téléchargé une fois depuis Hugging Face."""

    def __init__(self, modele: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer     # import tardif : dépendance lourde
        self.nom = f"sentence-transformers:{modele}"
        self._modele = SentenceTransformer(modele)
        # Les modèles e5 attendent ces préfixes ; les autres modèles n'en veulent pas.
        self._pref_passage, self._pref_requete = ("passage: ", "query: ") if "e5" in modele.lower() else ("", "")

    def _encoder(self, textes: list[str]) -> np.ndarray:
        v = self._modele.encode(textes, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def encoder_passages(self, textes):
        return self._encoder([self._pref_passage + t for t in textes])

    def encoder_requete(self, texte):
        return self._encoder([self._pref_requete + texte])[0]


class EmbedderHachage(Embedder):
    """Sac de mots + trigrammes de caractères « hachés » dans un vecteur de taille fixe.

    Sans dépendance ni téléchargement : sert aux tests et de solution de secours.
    Il ne comprend pas le sens (pas de synonymes ni de traduction), seulement les mots communs.
    """

    def __init__(self, dimension: int = 2048):
        self.nom = f"hachage:{dimension}"
        self.dimension = dimension

    @staticmethod
    def _normaliser(texte: str) -> str:
        texte = unicodedata.normalize("NFKD", texte.lower())
        texte = "".join(c for c in texte if not unicodedata.combining(c))
        return re.sub(r"[^\w]+", " ", texte)

    def _vecteur(self, texte: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for mot in self._normaliser(texte).split():
            if len(mot) < 3:
                continue
            traits = [mot] + [mot[i:i + 3] for i in range(len(mot) - 2)]
            for t in traits:
                h = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "little")
                v[h % self.dimension] += 1.0 if t == mot else 0.5
        n = np.linalg.norm(v)
        return v / n if n else v

    def encoder_passages(self, textes):
        return np.stack([self._vecteur(t) for t in textes]) if textes else np.zeros((0, self.dimension), np.float32)

    def encoder_requete(self, texte):
        return self._vecteur(texte)


def creer_embedder(config_embeddings: dict) -> Embedder:
    f = config_embeddings["fournisseur"]
    if f == "sentence-transformers":
        return EmbedderSentenceTransformers(config_embeddings.get("modele", "intfloat/multilingual-e5-small"))
    if f == "hachage":
        return EmbedderHachage()
    raise ValueError(f"Fournisseur d'embeddings inconnu : {f!r} (sentence-transformers | hachage)")
