"""
Data validation: are the leaves of folder A also present in folder B (copies, rotated/flipped
or augmented versions)? Uses the leaf embeddings of src/data/similarity.py (leaf crop + ResNet18,
rotation/flip invariant), NOT pHash (pHash matched different leaves on these photos).

Usage:
    python -m src.data.data_validation --a D:/dev/data_archive/source_1 --b data/raw/uguz --max 1000
    python -m src.data.data_validation --a data/processed/yolo/images/test --b data/processed/yolo/images/train

The "same leaf" threshold defaults to the one calibrated by the `detection_dataset` stage
(reports/data_validation/detection_dataset.json); use --seuil to force a value.

Outputs (reports/data_validation/<name>/):
    overlap.json        summary
    examples/*.jpg      most similar pairs (left = A, right = B) -> CHECK THEM BY EYE
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.similarity import calculer_embeddings

RACINE = Path(__file__).resolve().parents[2]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATS = RACINE / "reports" / "data_validation" / "detection_dataset.json"


def lister_images(dossier):
    return sorted(p for p in Path(dossier).rglob("*") if p.suffix.lower() in IMG_EXT)


def seuil_par_defaut():
    if STATS.exists():
        return json.loads(STATS.read_text(encoding="utf-8"))["seuil_similarite"]
    print("  [warning] no calibrated threshold yet (run `dvc repro detection_dataset`); using 0.95")
    return 0.95


def comparer(dossier_a, dossier_b, seuil, max_images=0, n_exemples=20, sortie=None):
    A, B = lister_images(dossier_a), lister_images(dossier_b)
    if not A or not B:
        raise SystemExit(f"Empty folder: A={len(A)} images, B={len(B)} images")
    if max_images and len(A) > max_images:
        rng = np.random.default_rng(0)
        A = [A[i] for i in sorted(rng.choice(len(A), max_images, replace=False))]

    print(f"A: {len(A)} images | B: {len(B)} images")
    EA, EB = calculer_embeddings(A), calculer_embeddings(B)
    S = EA @ EB.T
    j = S.argmax(1)
    sim = S[np.arange(len(A)), j]

    resume = {
        "dossier_a": str(dossier_a), "dossier_b": str(dossier_b), "n_a": len(A), "n_b": len(B),
        "seuil_similarite": seuil,
        "taux_meme_feuille": float(np.mean(sim >= seuil)),
        "similarite_mediane": float(np.median(sim)),
        "verdict": "fuite_ou_copies" if np.mean(sim >= seuil) > 0.05 else "ok",
    }
    if sortie:
        sortie = Path(sortie)
        (sortie / "examples").mkdir(parents=True, exist_ok=True)
        (sortie / "overlap.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
        for k, i in enumerate(np.argsort(-sim)[:n_exemples]):
            paire = np.hstack([cv2.resize(cv2.imread(str(A[i])), (224, 224)),
                               cv2.resize(cv2.imread(str(B[j[i]])), (224, 224))])
            cv2.putText(paire, f"sim {sim[i]:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(sortie / "examples" / f"{k:02d}_sim{sim[i]:.3f}.jpg"), paire)
    return resume


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="folder to check (e.g. test images)")
    p.add_argument("--b", required=True, help="reference folder (e.g. train images)")
    p.add_argument("--seuil", type=float, default=None, help="same-leaf cosine threshold")
    p.add_argument("--max", type=int, default=0, help="check only N random images of A (0 = all)")
    p.add_argument("--nom", default=None, help="report name (default: <a>_vs_<b>)")
    args = p.parse_args()

    seuil = args.seuil if args.seuil is not None else seuil_par_defaut()
    nom = args.nom or f"{Path(args.a).name}_vs_{Path(args.b).name}"
    sortie = RACINE / "reports" / "data_validation" / nom
    r = comparer(args.a, args.b, seuil, args.max, sortie=sortie)
    print(f"\nSame leaf (cosine >= {seuil:.4f}): {r['taux_meme_feuille']:.1%}")
    print(f"Median best similarity: {r['similarite_mediane']:.3f}  ->  verdict: {r['verdict']}")
    print(f"Check the pairs by eye: {sortie / 'examples'}")


if __name__ == "__main__":
    main()
