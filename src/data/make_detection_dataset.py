"""
DVC stage `detection_dataset`: build the image manifest for the detection pipeline.

    collect images from every source in configs/config.yaml (class = parent folder name,
    mapped with `detection.correspondance_classes`)
    -> remove byte-identical files (MD5)
    -> leaf embeddings (src/data/similarity.py: leaf crop + ResNet18, rotation/flip invariant)
    -> automatic threshold: high quantile of the similarity between an image and its nearest
       neighbour of ANOTHER class (= two different leaves); `regroupement.seuil` can force a value
    -> GROUP images above the threshold (same leaf / augmented copies)
    -> groups mixing classes are excluded (unreliable labels) and reported
    -> stratified train / val / test split BY GROUP (no copy of a test leaf in train)

Usage:
    python -m src.data.make_detection_dataset

Outputs:
    data/interim/detection_manifest.csv              path, source, class, group, split
    data/interim/embeddings.npy                      leaf embeddings (same order as the manifest input)
    reports/data_validation/label_conflicts.csv      images of mixed-class groups (excluded)
    reports/data_validation/detection_dataset.json   counts + threshold (DVC metric)
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.data.similarity import (calculer_embeddings, calibrer_seuil, grouper_par_similarite,
                                 voisins_plus_proches)

RACINE = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MANIFESTE = RACINE / "data" / "interim" / "detection_manifest.csv"
EMBEDDINGS = RACINE / "data" / "interim" / "embeddings.npy"
RAPPORTS = RACINE / "reports" / "data_validation"


def charger_yaml(chemin):
    with open(RACINE / chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collecter(sources, correspondance):
    """List images; the class comes from the parent folder name."""
    elements, ignores = [], Counter()
    correspondance = {k.lower(): v for k, v in correspondance.items()}
    for source in sources:
        racine_source = RACINE / source
        if not racine_source.exists():
            raise FileNotFoundError(f"Source not found: {racine_source} (did you run `dvc pull`?)")
        for chemin in sorted(racine_source.rglob("*")):
            if chemin.suffix.lower() not in IMG_EXT:
                continue
            classe = correspondance.get(chemin.parent.name.lower())
            if classe is None:
                ignores[chemin.parent.name] += 1
                continue
            elements.append({"chemin": chemin.relative_to(RACINE).as_posix(),
                             "source": Path(source).name, "classe": classe})
    for dossier, n in ignores.items():
        print(f"  [ignored] folder '{dossier}' not in correspondance_classes ({n} images)")
    return elements


def dedoublonner(elements):
    """Remove byte-identical files (MD5); keep the first one."""
    vus, gardes = set(), []
    for e in elements:
        h = hashlib.md5((RACINE / e["chemin"]).read_bytes()).hexdigest()
        if h not in vus:
            vus.add(h)
            gardes.append(e)
    return gardes


def exclure_groupes_mixtes(elements):
    """Groups with several classes = label conflict: excluded and returned for the report."""
    classes_par_groupe = defaultdict(set)
    for e in elements:
        classes_par_groupe[e["groupe"]].add(e["classe"])
    mixtes = {g for g, c in classes_par_groupe.items() if len(c) > 1}
    return [e for e in elements if e["groupe"] not in mixtes], [e for e in elements if e["groupe"] in mixtes]


def decouper(elements, train, val, test, seed):
    """Stratified split BY GROUP (key 'groupe'; each image is its own group if absent).
    Returns the same dicts with a 'split' key."""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train + val + test must equal 1 (got {total})")
    par_groupe = defaultdict(list)
    for i, e in enumerate(elements):
        par_groupe[e.get("groupe", f"img{i}")].append(e)
    groupes = list(par_groupe)
    classe_groupe = [par_groupe[g][0]["classe"] for g in groupes]

    g_train, g_reste, _, c_reste = train_test_split(groupes, classe_groupe, test_size=val + test,
                                                    random_state=seed, stratify=classe_groupe)
    g_val, g_test = train_test_split(g_reste, test_size=test / (val + test),
                                     random_state=seed, stratify=c_reste)
    resultat = []
    for nom, gs in (("train", g_train), ("val", g_val), ("test", g_test)):
        for g in gs:
            for e in par_groupe[g]:
                e["split"] = nom
                resultat.append(e)
    return resultat


def main():
    config = charger_yaml("configs/config.yaml")["detection"]
    params = charger_yaml("configs/param.yaml")["detection"]
    p_split, p_grp = params["decoupage"], params["regroupement"]

    print("1/5 Collecting...")
    elements = collecter(config["sources"], config["correspondance_classes"])
    n_collectees = len(elements)
    print(f"  {n_collectees} images: {dict(Counter(e['classe'] for e in elements))}")

    print("2/5 Removing byte-identical files...")
    elements = dedoublonner(elements)
    print(f"  {len(elements)} files left")

    print("3/5 Leaf embeddings (crop + ResNet18, 8 rotations/flips)...")
    E = calculer_embeddings([RACINE / e["chemin"] for e in elements], batch=p_grp["batch"])
    EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS, E)

    print("4/5 Calibrating threshold and grouping same-leaf images...")
    classes = [e["classe"] for e in elements]
    S, nn_meme, nn_autre = voisins_plus_proches(E, classes)
    seuil = (calibrer_seuil(nn_autre, p_grp["quantile_calibration"], p_grp["seuil_min"])
             if p_grp["seuil"] == "auto" else float(p_grp["seuil"]))
    for e, g in zip(elements, grouper_par_similarite(S, seuil)):
        e["groupe"] = g
    elements, exclus = exclure_groupes_mixtes(elements)
    tailles = Counter(e["groupe"] for e in elements)
    print(f"  threshold {seuil:.4f} | nearest other-class similarity: median {np.median(nn_autre):.3f}, "
          f"max {nn_autre.max():.3f} | nearest same-class: median {np.median(nn_meme):.3f}")
    print(f"  {len(tailles)} groups for {len(elements)} images | largest: {max(tailles.values())} | "
          f"{len({e['groupe'] for e in exclus})} mixed-class groups excluded ({len(exclus)} images)")

    print("5/5 Stratified split by group...")
    elements = decouper(elements, p_split["train"], p_split["val"], p_split["test"], p_split["seed"])

    RAPPORTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chemin", "source", "classe", "groupe", "split"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(elements)
    with open(RAPPORTS / "label_conflicts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["groupe", "chemin", "classe"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(exclus, key=lambda e: e["groupe"]))

    comptes = {s: dict(Counter(e["classe"] for e in elements if e["split"] == s))
               for s in ("train", "val", "test")}
    distribution = Counter(tailles.values())
    (RAPPORTS / "detection_dataset.json").write_text(json.dumps({
        "n_collectees": n_collectees,
        "n_images": len(elements),
        "n_groupes": len(tailles),
        "images_par_groupe_moyen": round(len(elements) / len(tailles), 2),
        "taille_max_groupe": max(tailles.values()),
        "groupes_par_taille": {str(k): distribution[k] for k in sorted(distribution)},
        "n_groupes_mixtes_exclus": len({e["groupe"] for e in exclus}),
        "n_images_exclues": len(exclus),
        "seuil_similarite": round(seuil, 4),
        "nn_autre_classe": {"median": round(float(np.median(nn_autre)), 4),
                            "p99": round(float(np.quantile(nn_autre, 0.99)), 4),
                            "max": round(float(nn_autre.max()), 4)},
        "nn_meme_classe": {"median": round(float(np.median(nn_meme)), 4),
                           "part_au_dessus_du_seuil": round(float(np.mean(nn_meme >= seuil)), 4)},
        "images_par_split": {s: sum(c.values()) for s, c in comptes.items()},
        "par_split": comptes,
    }, indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFESTE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
