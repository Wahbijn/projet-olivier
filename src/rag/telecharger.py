"""Télécharge les documents listés dans data/rag/sources.yaml vers data/rag/docs/.

    python -m src.rag.telecharger            # télécharge ce qui manque
    python -m src.rag.telecharger --force    # retélécharge tout
"""
import argparse

import httpx

from .chargeur import lire_metadonnees
from .config import charger_config_rag, chemin_projet
from urllib.parse import urlparse
ENTETES = {"User-Agent": "Mozilla/5.0 (projet-olivier RAG; usage pedagogique)"}
TAILLE_MAX = 30 * 1024 * 1024            # 30 Mo

def telecharger(url: str) -> tuple[bytes, str]:
    """Renvoie (contenu, extension)."""
    with httpx.Client(headers=ENTETES, follow_redirects=True, timeout=60) as client:
        rep = client.get(url)
        rep.raise_for_status()
    contenu = rep.content
    if len(contenu) > TAILLE_MAX:
        raise ValueError("fichier trop gros")
    if contenu[:5] == b"%PDF-":
        return contenu, ".pdf"
    est_html = "html" in rep.headers.get("content-type", "") or contenu.lstrip()[:15].lower().startswith((b"<!doctype", b"<html"))
    if est_html:
        if urlparse(url).path.lower().endswith(".pdf"):
            # L'URL promet un PDF mais le serveur renvoie une page web : blocage anti-robot,
            # page de consentement ou redirection. L'enregistrer polluerait l'index.
            raise ValueError("l'URL pointe vers un PDF mais le serveur a renvoye une page web (blocage anti-robot ?)")
        return contenu, ".html"
    raise ValueError(f"type de contenu non gere ({rep.headers.get('content-type')})")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="retélécharger même si le fichier existe")
    args = ap.parse_args()

    config = charger_config_rag()
    dossier = chemin_projet(config["docs_dir"])
    dossier.mkdir(parents=True, exist_ok=True)
    sources = lire_metadonnees(chemin_projet(config["sources"]))

    echecs = []
    for id_, info in sources.items():
        existants = list(dossier.glob(f"{id_}.*"))
        if existants and not args.force:
            print(f"= {id_}: déjà présent ({existants[0].name})")
            continue
        try:
            contenu, ext = telecharger(info["url"])
            (dossier / f"{id_}{ext}").write_bytes(contenu)
            print(f"+ {id_}: {len(contenu) // 1024} Ko ({ext})")
        except Exception as e:
            print(f"! {id_}: échec ({e})")
            echecs.append((id_, info["url"]))

    if echecs:
        print("\nÀ télécharger à la main (navigateur) dans", dossier, ":")
        for id_, url in echecs:
            print(f"  {id_}.pdf ou {id_}.html  <-  {url}")


if __name__ == "__main__":
    main()
