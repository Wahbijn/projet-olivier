"""Nettoyage et découpage des données (notebook, sections 1 à 4).

data/preprocessing/splits.csv est la RÉFÉRENCE versionnée dans Git : c'est le découpage
utilisé pour les résultats du README. Les chemins y sont relatifs à data/raw/ :
    data/raw/uguz/dataset/{train,test}/<classe>/...      (Kaggle habibulbasher01644/olive-leaf-image-dataset)
    data/raw/zeytin/{sağlam,hastalıklı}/...               (Kaggle serhathoca/zeytin)

Modes :
    python -m src.data.preprocessing                vérifie que toutes les images de splits.csv existent
    python -m src.data.preprocessing --regenerer    refait dédoublonnage + découpage (écrase splits.csv !)
    python -m src.data.preprocessing --migrer       convertit d'anciens chemins Kaggle absolus en chemins relatifs
"""
import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

from src.utils import RACINE, charger_config, charger_params

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
# préfixes Kaggle -> dossier local dans data/raw
PREFIXES_KAGGLE = {
    "/kaggle/input/datasets/habibulbasher01644/olive-leaf-image-dataset/": "uguz/",
    "/kaggle/input/datasets/serhathoca/zeytin/": "zeytin/",
}


# ------------------------------------------------------------------ migration
def migrer_chemin(chemin: str) -> str:
    for prefixe, local in PREFIXES_KAGGLE.items():
        if chemin.startswith(prefixe):
            return local + chemin[len(prefixe):]
    return chemin


def migrer(splits: pd.DataFrame) -> pd.DataFrame:
    splits = splits.copy()
    splits["chemin"] = splits["chemin"].map(migrer_chemin)
    # Zeytin « hastalıklı » = feuilles malades (œil de paon en réalité, cf. notebook section 8) :
    # on ne leur attribue pas une maladie précise, l'évaluation sur Zeytin est binaire.
    externe_malade = (splits["source"] == "zeytin") & (splits["classe"] != "healthy")
    splits.loc[externe_malade, "classe"] = "malade"
    return splits


# ------------------------------------------------------------------ vérification
def verifier(splits: pd.DataFrame, raw: Path) -> int:
    manquants = [c for c in splits["chemin"] if not (raw / c).is_file()]
    print(pd.crosstab(splits["split"], splits["classe"], margins=True), "\n")
    if manquants:
        print(f"❌ {len(manquants)} images introuvables dans {raw.relative_to(RACINE)}/, par ex. :")
        for c in manquants[:5]:
            print("   ", c)
        print("Voir README, section « Données », pour télécharger et placer les deux jeux.")
        return 1
    print(f"✅ Les {len(splits)} images de splits.csv sont présentes.")
    return 0


# ------------------------------------------------------------------ régénération
def inventaire(raw: Path, correspondance: dict) -> pd.DataFrame:
    lignes = []
    for chemin in sorted(raw.rglob("*")):                      # trié -> résultat déterministe
        if not chemin.is_file() or chemin.suffix.lower() not in EXTENSIONS:
            continue
        rel = chemin.relative_to(raw).as_posix()
        source = rel.split("/")[0]
        dossier = chemin.parent.name
        if dossier not in correspondance:
            raise SystemExit(f"Dossier de classe inconnu '{dossier}' ({rel}) : l'ajouter dans configs/config.yaml")
        split_origine = "test" if "/test/" in rel else ("train" if "/train/" in rel else "-")
        lignes.append({"chemin": rel, "source": source, "split_origine": split_origine,
                       "classe": correspondance[dossier]})
    return pd.DataFrame(lignes)


def empreinte16(chemin: Path) -> np.ndarray:
    """dHash 16x16 = 256 bits (notebook, cellule 6)."""
    with Image.open(chemin) as img:
        img.draft("L", (128, 128))
        img = img.convert("L").resize((17, 16))
    px = np.asarray(img, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def md5(chemin: Path) -> str:
    return hashlib.md5(chemin.read_bytes()).hexdigest()


def regenerer(raw: Path, config: dict, params: dict) -> pd.DataFrame:
    df = inventaire(raw, config["donnees"]["correspondance_classes"])
    print(f"{len(df)} images trouvées")
    print(pd.crosstab(df["source"], df["classe"], margins=True), "\n")

    chemins = [raw / c for c in df["chemin"]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        B = np.array(list(ex.map(empreinte16, chemins)), dtype=np.float32)
        df["md5"] = list(ex.map(md5, chemins))

    # Distance de Hamming entre toutes les paires, puis groupes connexes sous le seuil
    seuil = config["dedoublonnage"]["seuil_distance"]
    D = B @ (1 - B).T + (1 - B) @ B.T
    np.fill_diagonal(D, 0)
    proches = D <= seuil
    # les copies exactes (même MD5) sont aussi regroupées
    codes = pd.factorize(df["md5"])[0]
    proches |= codes[:, None] == codes[None, :]
    _, df["groupe_doublon"] = connected_components(proches, directed=False)

    taille = df.groupby("groupe_doublon")["chemin"].transform("count")
    nb_classes = df.groupby("groupe_doublon")["classe"].transform("nunique")
    uguz_dup = df[(df["source"] == "uguz") & (taille >= 2)]
    fuite = (uguz_dup.groupby("groupe_doublon")["split_origine"].nunique() > 1).sum()
    print(f"Groupes de doublons : {(df.groupby('groupe_doublon').size() >= 2).sum()} | "
          f"contradictoires (supprimés) : {df[nb_classes > 1]['groupe_doublon'].nunique()} | "
          f"fuite train/test d'origine Uğuz : {fuite}")

    propre = df[nb_classes == 1].drop_duplicates("groupe_doublon", keep="first").copy()

    d = params["decoupage"]
    uguz = propre[propre["source"] == "uguz"]
    train, temp = train_test_split(uguz, test_size=d["val"] + d["test"],
                                   stratify=uguz["classe"], random_state=d["seed"])
    val, test = train_test_split(temp, test_size=d["test"] / (d["val"] + d["test"]),
                                 stratify=temp["classe"], random_state=d["seed"])
    externe = propre[propre["source"] == "zeytin"]
    parts = []
    for nom, part in [("train", train), ("val", val), ("test", test), ("externe", externe)]:
        part = part.copy()
        part["split"] = nom
        parts.append(part)
    splits = pd.concat(parts)[["chemin", "source", "classe", "split"]]
    return migrer(splits)            # même convention d'étiquettes Zeytin


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    groupe = ap.add_mutually_exclusive_group()
    groupe.add_argument("--regenerer", action="store_true")
    groupe.add_argument("--migrer", action="store_true")
    args = ap.parse_args()

    config, params = charger_config(), charger_params()
    raw = RACINE / config["donnees"]["raw"]
    fichier = RACINE / config["donnees"]["splits"]

    if args.migrer:
        splits = migrer(pd.read_csv(fichier))
        splits.to_csv(fichier, index=False)
        print(f"Chemins convertis -> {fichier.relative_to(RACINE)}")
        return
    if args.regenerer:
        splits = regenerer(raw, config, params)
        splits.to_csv(fichier, index=False)
        print(f"\nNouveau découpage écrit dans {fichier.relative_to(RACINE)} :")
    else:
        splits = pd.read_csv(fichier)
    sys.exit(verifier(splits, raw))


if __name__ == "__main__":
    main()
