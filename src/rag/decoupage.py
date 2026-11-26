"""Découpage des documents en passages (chunks) de taille raisonnable, avec chevauchement."""
import re
from dataclasses import dataclass

from .chargeur import Document


@dataclass
class Passage:
    id: str               # "<document>#<numéro>"
    document_id: str
    titre: str
    url: str | None
    texte: str


def _part_de_lettres(passage: str) -> float:
    return sum(c.isalpha() for c in passage) / max(len(passage), 1)


# « Keifer HH (1975) », « Ullah Z (2022) », « Hughes H. (1953) » : motif typique d'une liste de références.
# (« Smith et al. (2001) » dans une phrase ne correspond pas : « al. » est en minuscules.)
_REFERENCE = re.compile(r"\b[A-ZÀ-Þİ]{1,3}\.?\s*\((?:19|20)\d{2}[a-z]?\)")


def _inutile(passage: str) -> bool:
    """Passage qui ne peut pas répondre à une question : trop court, tableau de chiffres, liste de références."""
    if len(passage) < 60 or _part_de_lettres(passage) < 0.6:
        return True
    return len(_REFERENCE.findall(passage)) >= 3


def _phrases(texte: str) -> list[str]:
    """Coupe d'abord par paragraphes, puis par phrases les paragraphes trop longs."""
    morceaux = []
    for para in re.split(r"\n\s*\n", texte):
        para = " ".join(para.split())
        if not para:
            continue
        morceaux.append(para)
    return morceaux


def decouper(texte: str, taille: int = 900, chevauchement: int = 150) -> list[str]:
    """Regroupe des paragraphes jusqu'à `taille` caractères ; les blocs trop longs sont coupés
    sur la ponctuation. Chaque passage reprend la fin du précédent (`chevauchement`)."""
    if chevauchement >= taille:
        raise ValueError("Le chevauchement doit être inférieur à la taille.")

    unites = []
    for para in _phrases(texte):
        if len(para) <= taille:
            unites.append(para)
        else:  # paragraphe géant : on le coupe en phrases
            unites.extend(s for s in re.split(r"(?<=[.!?؟])\s+", para) if s)

    passages, courant = [], ""
    for u in unites:
        while len(u) > taille:                       # phrase géante : coupe dure
            coupe = u.rfind(" ", 0, taille)
            if coupe <= 0:                           # pas d'espace : coupe dure
                coupe = taille
            tete, u = u[:coupe], u[coupe:].lstrip()
            if courant:
                passages.append(courant)
                courant = ""
            passages.append(tete)
        if not u:
            continue
        if courant and len(courant) + 1 + len(u) > taille:
            passages.append(courant)
            queue = courant[-chevauchement:]
            queue = queue[queue.find(" ") + 1:] if " " in queue else queue   # pas de mot coupé
            courant = (queue + " " + u).strip()
        else:
            courant = (courant + " " + u).strip()
    if courant:
        passages.append(courant)
    return [p for p in passages if not _inutile(p)]


def decouper_documents(documents: list[Document], taille: int = 900, chevauchement: int = 150) -> list[Passage]:
    resultat = []
    for doc in documents:
        for i, texte in enumerate(decouper(doc.texte, taille, chevauchement)):
            resultat.append(Passage(f"{doc.id}#{i}", doc.id, doc.titre, doc.url, texte))
    return resultat
