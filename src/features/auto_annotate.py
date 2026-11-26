"""
DVC stage `annotation`: automatic YOLO bounding boxes (method of the OLIVE-CAD paper).

    Gaussian blur -> HSV -> leaf colour mask -> contours -> score each contour
    (area, solidity, centredness, aspect) -> best contour -> padded box -> YOLO label.
    No leaf found -> whole-image box, flagged as `fallback` (to review by hand).

Usage:
    python -m src.features.auto_annotate

Inputs:  data/interim/detection_manifest.csv, configs/config.yaml, configs/param.yaml
Outputs:
    data/processed/yolo/images/{train,val,test}/*.jpg
    data/processed/yolo/labels/{train,val,test}/*.txt
    data/processed/yolo/data.yaml
    data/processed/yolo/annotations.csv              name, original path, class, split, box, fallback
    reports/annotation_review/<class|fallback>/*.jpg overlays for manual review
    reports/annotation_summary.json                  fallback counts (DVC metric)
"""
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

RACINE = Path(__file__).resolve().parents[2]
MANIFESTE = RACINE / "data" / "interim" / "detection_manifest.csv"
SORTIE = RACINE / "data" / "processed" / "yolo"
REVUE = RACINE / "reports" / "annotation_review"


def charger_yaml(chemin):
    with open(RACINE / chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def boite_feuille(img, marge=0.05, hsv_min=(10, 30, 25), hsv_max=(95, 255, 255), aire_min=0.02):
    """Return ((cx, cy, w, h) normalised to 0-1, fallback_used)."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, tuple(hsv_min), tuple(hsv_max))
    noyau = np.ones((5, 5), np.uint8)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau, iterations=2)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau, iterations=1)
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    meilleur, meilleur_score = None, -1.0
    for c in contours:
        aire = cv2.contourArea(c)
        if aire < aire_min * H * W:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        solidite = aire / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        cx, cy = x + bw / 2, y + bh / 2
        centrage = 1 - (abs(cx - W / 2) / (W / 2) + abs(cy - H / 2) / (H / 2)) / 2
        ratio = min(bw, bh) / max(bw, bh)
        score = 0.45 * (aire / (H * W)) + 0.2 * solidite + 0.3 * centrage + 0.05 * ratio
        if score > meilleur_score:
            meilleur, meilleur_score = (x, y, bw, bh), score

    if meilleur is None:
        return (0.5, 0.5, 1.0, 1.0), True
    x, y, bw, bh = meilleur
    px, py = bw * marge, bh * marge
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + bw + px), min(H, y + bh + py)
    return ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H), False


def dessiner(img, boite, texte, fallback):
    H, W = img.shape[:2]
    cx, cy, bw, bh = boite
    vis = img.copy()
    couleur = (0, 0, 255) if fallback else (0, 255, 0)
    cv2.rectangle(vis, (int((cx - bw / 2) * W), int((cy - bh / 2) * H)),
                  (int((cx + bw / 2) * W), int((cy + bh / 2) * H)), couleur, 2)
    cv2.putText(vis, texte, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def main():
    classes = charger_yaml("configs/config.yaml")["detection"]["classes"]
    p = charger_yaml("configs/param.yaml")["detection"]["annotation"]

    for dossier in (SORTIE, REVUE):
        if dossier.exists():
            shutil.rmtree(dossier)

    with open(MANIFESTE, encoding="utf-8") as f:
        manifeste = list(csv.DictReader(f))

    restant_revue = defaultdict(lambda: p["revue_par_classe"])
    fallbacks, lignes = Counter(), []
    for n, e in enumerate(manifeste):
        img = cv2.imread(str(RACINE / e["chemin"]))
        if img is None:
            print(f"  [unreadable] {e['chemin']}")
            continue
        boite, fb = boite_feuille(img, p["marge"], p["hsv_min"], p["hsv_max"], p["aire_min"])
        nom = f"{e['classe']}_{n:06d}"
        for sous in ("images", "labels"):
            (SORTIE / sous / e["split"]).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SORTIE / "images" / e["split"] / f"{nom}.jpg"), img)
        (SORTIE / "labels" / e["split"] / f"{nom}.txt").write_text(
            f"{classes.index(e['classe'])} " + " ".join(f"{v:.6f}" for v in boite) + "\n")
        fallbacks[e["classe"]] += int(fb)
        lignes.append([nom, e["chemin"], e["classe"], e["split"], *[f"{v:.6f}" for v in boite], int(fb)])

        if fb or restant_revue[e["classe"]] > 0:
            sous = "fallback" if fb else e["classe"]
            (REVUE / sous).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(REVUE / sous / f"{nom}.jpg"),
                        dessiner(img, boite, e["classe"] + (" FALLBACK" if fb else ""), fb))
            if not fb:
                restant_revue[e["classe"]] -= 1

    with open(SORTIE / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nom", "chemin_original", "classe", "split", "cx", "cy", "w", "h", "fallback"])
        w.writerows(lignes)
    # No absolute `path` here (machine-independent, safe to version);
    # src/models/train_yolo.py injects the absolute path at run time.
    (SORTIE / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)), encoding="utf-8")
    REVUE.mkdir(parents=True, exist_ok=True)
    (RACINE / "reports" / "annotation_summary.json").write_text(json.dumps(
        {"n_images": len(lignes), "fallback_par_classe": dict(fallbacks),
         "taux_fallback": sum(fallbacks.values()) / max(len(lignes), 1)}, indent=2), encoding="utf-8")
    print(f"{len(lignes)} images annotated | fallback boxes: {dict(fallbacks)}")
    print(f"Check the overlays in {REVUE.relative_to(RACINE)}/ before training.")


if __name__ == "__main__":
    main()
