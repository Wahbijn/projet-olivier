"""Tests for the detection data pipeline (no dataset needed: synthetic images)."""
import cv2
import numpy as np
import pytest

from src.data.make_detection_dataset import decouper
from src.data.similarity import calculer_embeddings, grouper_par_similarite
from src.features.auto_annotate import boite_feuille


def feuille_synthetique(blanche=False):
    img = np.full((224, 224, 3), 235, np.uint8)
    if not blanche:
        cv2.ellipse(img, (112, 112), (90, 30), 30, 0, 360, (40, 150, 40), -1)   # green leaf
        cv2.circle(img, (120, 110), 8, (30, 60, 90), -1)                          # brown spot
    return img


def test_box_covers_the_leaf():
    (cx, cy, w, h), fallback = boite_feuille(feuille_synthetique())
    assert not fallback
    assert 0.4 < cx < 0.6 and 0.4 < cy < 0.6          # centred
    assert 0.5 < w <= 1.0 and 0.3 < h <= 1.0           # big enough to contain the leaf
    assert all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))  # YOLO format


def test_no_leaf_gives_fallback_box():
    boite, fallback = boite_feuille(feuille_synthetique(blanche=True))
    assert fallback
    assert boite == (0.5, 0.5, 1.0, 1.0)


def test_split_is_stratified_and_complete():
    elements = [{"chemin": f"{c}_{i}.jpg", "classe": c} for c in ("healthy", "aculus", "peacock_spot")
                for i in range(100)]
    resultat = decouper(elements, 0.7, 0.2, 0.1, seed=42)
    assert len(resultat) == 300
    assert len({e["chemin"] for e in resultat}) == 300          # no image in two splits
    for classe in ("healthy", "aculus", "peacock_spot"):
        n_test = sum(e["split"] == "test" and e["classe"] == classe for e in resultat)
        assert 9 <= n_test <= 11                                # ~10 % of 100, per class (rounding)


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        decouper([{"chemin": "a", "classe": "x"}] * 10, 0.7, 0.2, 0.2, seed=0)


def test_copies_of_a_leaf_stay_in_the_same_split():
    """All images of a group (augmented copies of one leaf) must land in a single split."""
    elements = [{"chemin": f"{c}_{g}_{k}.jpg", "classe": c, "groupe": f"{c}{g}"}
                for c in ("healthy", "aculus", "peacock_spot") for g in range(40) for k in range(3)]
    resultat = decouper(elements, 0.7, 0.2, 0.1, seed=0)
    splits_par_groupe = {}
    for e in resultat:
        splits_par_groupe.setdefault(e["groupe"], set()).add(e["split"])
    assert all(len(s) == 1 for s in splits_par_groupe.values())
    assert {e["split"] for e in resultat} == {"train", "val", "test"}


def test_grouping_is_transitive_and_respects_threshold():
    S = np.array([[1.0, 0.98, 0.10, 0.10],
                  [0.98, 1.0, 0.97, 0.10],
                  [0.10, 0.97, 1.0, 0.20],
                  [0.10, 0.10, 0.20, 1.0]])
    g = grouper_par_similarite(S, 0.96)
    assert g[0] == g[1] == g[2]          # 0~1 and 1~2 -> same group
    assert g[3] != g[0]


def test_embedding_is_rotation_and_flip_invariant(tmp_path):
    """Same leaf rotated/flipped -> near-identical embedding (random weights: no download in CI)."""
    img = feuille_synthetique()
    chemins = []
    for nom, variante in (("orig", img), ("rot", np.rot90(img).copy()), ("flip", cv2.flip(img, 1))):
        chemins.append(tmp_path / f"{nom}.png")
        cv2.imwrite(str(chemins[-1]), variante)
    E = calculer_embeddings(chemins, device="cpu", pretrained=False)
    assert E.shape[0] == 3
    assert float(E[0] @ E[1]) > 0.98 and float(E[0] @ E[2]) > 0.98
