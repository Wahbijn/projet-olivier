"""
Data validation: detect duplicated / augmented copies between two image folders
(perceptual hash, including the 8 rotations/flips).

Used to check:
  * whether a dataset is an augmented copy of another (e.g. source_1 vs uguz)
  * leakage between the train and test folders of a dataset

Usage:
    python -m src.data.data_validation --a data/raw/source_1 --b data/raw/uguz --max 1000

Outputs (reports/data_validation/<name>/):
    overlap.json        summary (rates, median distance, verdict)
    examples/*.jpg      closest pairs side by side (left = A, right = B)
"""
import argparse
import json
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

RACINE = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

SEUIL_IDENTIQUE = 6      # <= 6 bits of 64: same image
SEUIL_PROCHE = 12        # 7-12: maybe the same leaf (check by eye)


def lister_images(dossier):
    return sorted(p for p in Path(dossier).rglob("*") if p.suffix.lower() in IMG_EXT)


def hash64(img: Image.Image) -> np.uint64:
    return np.uint64(int(str(imagehash.phash(img)), 16))


def variantes(img: Image.Image):
    """The 8 rotations/flips (common augmentations)."""
    for miroir in (False, True):
        base = img.transpose(Image.FLIP_LEFT_RIGHT) if miroir else img
        for k in range(4):
            yield base.rotate(90 * k, expand=True)


def distances_hamming(h: np.uint64, tableau: np.ndarray) -> np.ndarray:
    return _POPCOUNT[(tableau ^ h).view(np.uint8).reshape(-1, 8)].sum(1)


def plus_proche(img: Image.Image, hashes_b: np.ndarray):
    """(min distance, index in B) over the 8 variants of img."""
    d_min, j_min = 64, -1
    for v in variantes(img):
        d = distances_hamming(hash64(v), hashes_b)
        j = int(d.argmin())
        if d[j] < d_min:
            d_min, j_min = int(d[j]), j
    return d_min, j_min


def comparer(dossier_a, dossier_b, max_images=0, n_exemples=20, sortie=None):
    A, B = lister_images(dossier_a), lister_images(dossier_b)
    if not A or not B:
        raise SystemExit(f"Empty folder: A={len(A)} images, B={len(B)} images")
    if max_images and len(A) > max_images:
        rng = np.random.default_rng(0)
        A = [A[i] for i in rng.choice(len(A), max_images, replace=False)]

    hashes_b = np.array([hash64(Image.open(f).convert("RGB")) for f in B], dtype=np.uint64)
    resultats = []
    for i, f in enumerate(A, 1):
        d, j = plus_proche(Image.open(f).convert("RGB"), hashes_b)
        resultats.append((d, f, B[j]))
        if i % 500 == 0:
            print(f"  {i}/{len(A)}")

    d = np.array([r[0] for r in resultats])
    resume = {
        "dossier_a": str(dossier_a), "dossier_b": str(dossier_b),
        "n_a": len(A), "n_b": len(B),
        "taux_identiques": float(np.mean(d <= SEUIL_IDENTIQUE)),
        "taux_proches": float(np.mean(d <= SEUIL_PROCHE)),
        "distance_mediane": float(np.median(d)),
        "verdict": "copies_ou_augmentations" if np.mean(d <= SEUIL_IDENTIQUE) > 0.2 else "peu_de_chevauchement",
    }

    if sortie:
        sortie = Path(sortie)
        (sortie / "examples").mkdir(parents=True, exist_ok=True)
        (sortie / "overlap.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
        for k, (dist, fa, fb) in enumerate(sorted(resultats, key=lambda r: r[0])[:n_exemples]):
            paire = np.hstack([cv2.resize(cv2.imread(str(fa)), (224, 224)),
                               cv2.resize(cv2.imread(str(fb)), (224, 224))])
            cv2.putText(paire, f"dist {dist}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(sortie / "examples" / f"{k:02d}_dist{dist}.jpg"), paire)
    return resume


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="folder to check (e.g. the augmented set)")
    p.add_argument("--b", required=True, help="reference folder (e.g. the original set)")
    p.add_argument("--max", type=int, default=0, help="check only N random images of A (0 = all)")
    p.add_argument("--nom", default=None, help="report name (default: <a>_vs_<b>)")
    args = p.parse_args()

    nom = args.nom or f"{Path(args.a).name}_vs_{Path(args.b).name}"
    sortie = RACINE / "reports" / "data_validation" / nom
    r = comparer(args.a, args.b, args.max, sortie=sortie)
    print(f"\nSame image (dist <= {SEUIL_IDENTIQUE}):  {r['taux_identiques']:.1%}")
    print(f"Maybe same leaf (dist <= {SEUIL_PROCHE}): {r['taux_proches']:.1%}")
    print(f"Median distance: {r['distance_mediane']:.0f}  ->  verdict: {r['verdict']}")
    print(f"Report: {sortie}")


if __name__ == "__main__":
    main()
