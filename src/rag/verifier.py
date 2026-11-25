"""Affiche, pour chaque document téléchargé, sa taille et le début de son texte.

    python -m src.rag.verifier
"""
from .chargeur import charger_documents
from .config import charger_config_rag, chemin_projet

MOTS_SUSPECTS = ("captcha", "verify you are human", "preparing to download", "access denied",
                 "enable javascript", "are you a robot", "just a moment", "cookies")


def main():
    config = charger_config_rag()
    docs = charger_documents(chemin_projet(config["docs_dir"]), chemin_projet(config["sources"]))
    if not docs:
        raise SystemExit("Aucun document exploitable dans data/rag/docs.")
    for d in docs:
        debut = " ".join(d.texte.split())
        suspect = len(d.texte) < 3000 or any(m in debut[:600].lower() for m in MOTS_SUSPECTS)
        print(f"{'⚠ SUSPECT' if suspect else 'OK       '}  {d.id}  ({len(d.texte)} caractères)")
        print(f"           {debut[:220]}…\n")
    print("Un document SUSPECT n'est pas forcément faux (une fiche courte est normale), mais lis son début :")
    print("si ce n'est pas le contenu attendu, supprime le fichier et télécharge-le avec ton navigateur.")


if __name__ == "__main__":
    main()