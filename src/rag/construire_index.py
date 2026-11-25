"""Construit l'index RAG à partir des documents de data/rag/docs.

    python -m src.rag.construire_index
"""
from collections import Counter

from .chargeur import charger_documents
from .config import charger_config_rag, chemin_projet
from .decoupage import decouper_documents
from .embeddings import creer_embedder
from .index import Index


def main():
    config = charger_config_rag()
    dossier = chemin_projet(config["docs_dir"])
    print(f"Lecture des documents dans {dossier} ...")
    documents = charger_documents(dossier, chemin_projet(config["sources"]))
    if not documents:
        raise SystemExit("Aucun document exploitable. Lancer d'abord : python -m src.rag.telecharger "
                         "(ou déposer des PDF / HTML dans ce dossier).")

    d = config["decoupage"]
    passages = decouper_documents(documents, d["taille"], d["chevauchement"])
    compte = Counter(p.document_id for p in passages)
    for doc in documents:
        print(f"  {doc.id:<35} {len(doc.texte):>8} caractères -> {compte[doc.id]:>3} passages")

    print(f"Calcul des embeddings de {len(passages)} passages ({config['embeddings']['modele']}) ...")
    embedder = creer_embedder(config["embeddings"])
    index = Index.construire(passages, embedder)
    index.sauvegarder(chemin_projet(config["index_dir"]))
    print(f"Index enregistré dans {chemin_projet(config['index_dir'])} ({len(index)} passages).")


if __name__ == "__main__":
    main()
