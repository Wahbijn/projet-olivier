"""Tests du prétraitement et du découpage (lancement : pytest)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.preprocessing import migrer, migrer_chemin
from src.features.feature_extraction import preparer_image, transformations
from src.models.architectures import creer_modele
from src.utils import RACINE, charger_config


def test_preparer_image_carre_et_taille():
    img = Image.new("RGB", (800, 600), (200, 200, 200))
    a = preparer_image(img, 224)
    assert a.shape == (224, 224, 3) and a.dtype == np.uint8
    # le padding reprend la couleur du fond (bandes haut/bas d'une image paysage)
    assert tuple(a[0, 112]) == (200, 200, 200)


def test_preparer_image_accepte_niveaux_de_gris_et_rgba():
    for mode in ("L", "RGBA"):
        assert preparer_image(Image.new(mode, (300, 500)), 224).shape == (224, 224, 3)


def test_transformations_sortie_normalisee():
    train, evaluation = transformations(224)
    x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    for t in (train, evaluation):
        y = t(x)
        assert tuple(y.shape) == (3, 224, 224)


def test_migration_chemins_kaggle():
    assert migrer_chemin("/kaggle/input/datasets/habibulbasher01644/olive-leaf-image-dataset/"
                         "dataset/train/Healthy/B-171.JPG") == "uguz/dataset/train/Healthy/B-171.JPG"
    assert migrer_chemin("/kaggle/input/datasets/serhathoca/zeytin/sağlam/a.png") == "zeytin/sağlam/a.png"
    df = pd.DataFrame({"chemin": ["/kaggle/input/datasets/serhathoca/zeytin/hastalıklı/b.png"],
                       "source": ["zeytin"], "classe": ["aculus"], "split": ["externe"]})
    assert migrer(df).loc[0, "classe"] == "malade"


@pytest.fixture(scope="module")
def splits():
    return pd.read_csv(RACINE / charger_config()["donnees"]["splits"])


def test_splits_chemins_relatifs(splits):
    assert not splits["chemin"].str.startswith("/").any()
    assert set(splits["chemin"].str.split("/").str[0]) == {"uguz", "zeytin"}


def test_splits_sans_fuite(splits):
    # chaque image n'apparaît qu'une fois, et Zeytin reste un test externe
    assert not splits["chemin"].duplicated().any()
    assert set(splits.loc[splits["source"] == "zeytin", "split"]) == {"externe"}
    assert set(splits.loc[splits["source"] == "uguz", "split"]) == {"train", "val", "test"}


def test_splits_classes_coherentes(splits):
    classes = set(charger_config()["classes"])
    uguz = splits[splits["source"] == "uguz"]
    assert set(uguz["classe"]) == classes
    assert set(splits.loc[splits["source"] == "zeytin", "classe"]) <= {"healthy", "malade"}


@pytest.mark.parametrize("architecture", ["resnet50", "efficientnet_b0"])
def test_architectures_trois_sorties(architecture):
    import torch
    m = creer_modele(architecture, 3, pretrained=False).eval()
    with torch.no_grad():
        assert tuple(m(torch.zeros(1, 3, 224, 224)).shape) == (1, 3)
