"""Affiche les passages retrouvés et leurs scores, SANS appeler le LLM.

Sert à calibrer `score_min` : tester une question pertinente et une question hors sujet,
puis placer le seuil entre les deux.

    python -m src.rag.diagnostic "Comment reconnaître l'œil de paon ?"
    python -m src.rag.diagnostic "Quelle est la capitale de la France ?"
    python -m src.rag.diagnostic "Que faire ?" --classe aculus
"""
import argparse

from .config import charger_config_rag, chemin_projet
from .embeddings import creer_embedder
from .index import Index
from .pipeline import MOTS_CLES_CLASSE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--classe", choices=list(MOTS_CLES_CLASSE))
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    config = charger_config_rag()
    index = Index.charger(chemin_projet(config["index_dir"]))
    embedder = creer_embedder(config["embeddings"])
    requete = args.question + (f" {MOTS_CLES_CLASSE[args.classe]}" if args.classe else "")
    seuil = config["recherche"]["score_min"][config["embeddings"]["fournisseur"]]

    brut = index.chercher(embedder.encoder_requete(args.question), k=1)[0].score
    verdict = "ACCEPTÉE" if brut >= seuil else "REFUSÉE (hors sujet)"
    print(f"Seuil actuel (score_min) : {seuil}")
    print(f"PERTINENCE de la question seule : {brut:.3f}  ->  {verdict}\n")
    for r in index.chercher(embedder.encoder_requete(requete), k=args.k):
        marque = "•"
        apercu = " ".join(r.passage.texte.split())[:110]
        print(f"{marque} {r.score:.3f}  {r.passage.id:<32} {apercu}…")


if __name__ == "__main__":
    main()
