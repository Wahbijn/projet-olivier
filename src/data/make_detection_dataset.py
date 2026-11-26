"""
DVC stage `detection_dataset`: build the image manifest for the detection pipeline.

    collect images from every source in configs/config.yaml (class = parent folder name,
    mapped with `detection.correspondance_classes`)
    -> remove duplicates ACROSS sources (perceptual hash) BEFORE splitting
    -> stratified train / val / test split (configs/param.yaml: detection.decoupage)

Usage:
    python -m src.data.make_detection_dataset

Outputs:
    data/interim/detection_manifest.csv              path, source, class, split
    reports/data_validation/label_conflicts.csv      same image found with 2 different classes
    reports/data_validation/detection_dataset.json   counts (DVC metric)
"""
import csv
import json
from collections import Counter
from pathlib import Path

import imagehash
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MANIFESTE = RACINE / "data" / "interim" / "detection_manifest.csv"
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
    """Keep the first image of each perceptual hash; report class conflicts."""
    vus, gardes, conflits = {}, [], []
    for e in elements:
        try:
            h = str(imagehash.phash(Image.open(RACINE / e["chemin"]).convert("RGB")))
        except Exception as err:
            print(f"  [unreadable] {e['chemin']}: {err}")
            continue
        if h in vus:
            premier = vus[h]
            if premier["classe"] != e["classe"]:
                conflits.append((premier["chemin"], premier["classe"], e["chemin"], e["classe"]))
            continue
        vus[h] = e
        gardes.append(e)
    return gardes, conflits


def decouper(elements, train, val, test, seed):
    """Stratified split; returns the same dicts with a 'split' key."""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train + val + test must equal 1 (got {total})")
    a, reste = train_test_split(elements, test_size=val + test, random_state=seed,
                                stratify=[e["classe"] for e in elements])
    v, t = train_test_split(reste, test_size=test / (val + test), random_state=seed,
                            stratify=[e["classe"] for e in reste])
    for nom, groupe in (("train", a), ("val", v), ("test", t)):
        for e in groupe:
            e["split"] = nom
    return a + v + t


def main():
    config = charger_yaml("configs/config.yaml")["detection"]
    params = charger_yaml("configs/param.yaml")["detection"]["decoupage"]

    print("1/3 Collecting...")
    elements = collecter(config["sources"], config["correspondance_classes"])
    print(f"  {len(elements)} images: {dict(Counter(e['classe'] for e in elements))}")

    print("2/3 Removing duplicates across sources...")
    elements, conflits = dedoublonner(elements)
    print(f"  {len(elements)} unique images, {len(conflits)} class conflicts")

    print("3/3 Stratified split...")
    elements = decouper(elements, params["train"], params["val"], params["test"], params["seed"])

    MANIFESTE.parent.mkdir(parents=True, exist_ok=True)
    RAPPORTS.mkdir(parents=True, exist_ok=True)
    with open(MANIFESTE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chemin", "source", "classe", "split"])
        w.writeheader()
        w.writerows(elements)
    with open(RAPPORTS / "label_conflicts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chemin_a", "classe_a", "chemin_b", "classe_b"])
        w.writerows(conflits)
    comptes = {s: dict(Counter(e["classe"] for e in elements if e["split"] == s))
               for s in ("train", "val", "test")}
    (RAPPORTS / "detection_dataset.json").write_text(json.dumps(
        {"n_images": len(elements), "n_conflits_classe": len(conflits), "par_split": comptes},
        indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFESTE.relative_to(RACINE)}  |  {comptes}")


if __name__ == "__main__":
    main()
