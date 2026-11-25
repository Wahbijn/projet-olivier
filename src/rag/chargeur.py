"""Lecture des documents (PDF, HTML, Markdown, texte) en texte brut."""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".txt"}


@dataclass
class Document:
    id: str            # identifiant stable (nom du fichier sans extension)
    titre: str
    url: str | None
    langue: str | None
    texte: str


def _nettoyer(texte: str) -> str:
    texte = texte.replace("\x00", " ").replace("\xa0", " ")
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


TITRE_BIBLIOGRAPHIE = re.compile(
    r"^\s*(references?(\s+cited)?|références(\s+bibliographiques)?|bibliographie|bibliography|"
    r"literature\s+cited|bibliografia|referencias|kaynaklar)\s*:?\s*$", re.I | re.M)


def nettoyer_document(texte: str) -> str:
    """Retire ce qui pollue la recherche : en-têtes / pieds de page répétés à chaque page, et la
    bibliographie finale (une liste de références ressemble à tout, mais ne répond à rien)."""
    lignes = texte.split("\n")

    def cle(ligne: str) -> str:                       # même ligne, hors numéro de page
        return re.sub(r"\d+", "", ligne).strip().lower()

    def candidate(ligne: str) -> bool:
        return 15 <= len(ligne.strip()) <= 250

    repetitions = Counter(cle(l) for l in lignes if candidate(l))
    texte = "\n".join(l for l in lignes if not (candidate(l) and repetitions[cle(l)] >= 3))

    for m in TITRE_BIBLIOGRAPHIE.finditer(texte):     # premier titre « References » dans la fin du document
        if m.start() > 0.4 * len(texte):
            return texte[:m.start()].strip()
    return texte.strip()


def lire_pdf(chemin: Path) -> str:
    from pypdf import PdfReader
    lecteur = PdfReader(str(chemin))
    return "\n\n".join((page.extract_text() or "") for page in lecteur.pages)


def lire_html(chemin: Path) -> str:
    from bs4 import BeautifulSoup
    soupe = BeautifulSoup(chemin.read_bytes(), "html.parser")
    for balise in soupe(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        balise.decompose()
    principal = soupe.find("main") or soupe.find("article") or soupe.body or soupe
    return principal.get_text("\n")


def lire_fichier(chemin: Path) -> str:
    ext = chemin.suffix.lower()
    if ext == ".pdf":
        brut = lire_pdf(chemin)
    elif ext in (".html", ".htm"):
        brut = lire_html(chemin)
    else:
        brut = chemin.read_text(encoding="utf-8", errors="replace")
    return _nettoyer(brut)


def lire_metadonnees(chemin_sources: Path) -> dict[str, dict]:
    """sources.yaml -> {id: {titre, url, langue, ...}}"""
    if not chemin_sources.exists():
        return {}
    with open(chemin_sources, encoding="utf-8") as f:
        contenu = yaml.safe_load(f) or {}
    return {s["id"]: s for s in contenu.get("sources", [])}


def charger_documents(dossier: Path, chemin_sources: Path | None = None) -> list[Document]:
    """Lit tous les documents du dossier. Les fichiers vides ou illisibles sont ignorés (avec un message)."""
    meta = lire_metadonnees(chemin_sources) if chemin_sources else {}
    documents = []
    for chemin in sorted(dossier.glob("*")):
        if chemin.suffix.lower() not in EXTENSIONS:
            continue
        try:
            texte = lire_fichier(chemin)
        except Exception as e:                      # PDF corrompu, encodage, etc.
            print(f"  ! {chemin.name} ignoré : {e}")
            continue
        texte = nettoyer_document(texte)
        if len(texte) < 200:
            print(f"  ! {chemin.name} ignoré : texte vide ou trop court (PDF scanné ?)")
            continue
        info = meta.get(chemin.stem, {})
        documents.append(Document(
            id=chemin.stem,
            titre=info.get("titre", chemin.stem),
            url=info.get("url"),
            langue=info.get("langue"),
            texte=texte,
        ))
    return documents
