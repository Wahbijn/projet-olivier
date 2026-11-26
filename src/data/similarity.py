"""
Leaf similarity with a pretrained CNN (replaces pHash, which is too coarse for leaf photos:
every image is "a dark elongated blob on a white background", so different leaves matched).

    image -> crop to the leaf (box from src.features.auto_annotate.boite_feuille)
          -> square padding -> 224x224
          -> ResNet18 (ImageNet) features of the 8 rotations/flips, averaged
             (=> the same leaf rotated or flipped gets the same descriptor)
          -> L2-normalised embedding; similarity = cosine (dot product)

Threshold calibration (`calibrer_seuil`): the nearest neighbour of an image in ANOTHER class
is, by definition, a different leaf. A high quantile of those similarities is therefore the
"most similar two different leaves can look"; same-leaf pairs must score above it.
"""
from collections import defaultdict

import cv2
import numpy as np

from src.features.auto_annotate import boite_feuille

MOYENNE = np.array([0.485, 0.456, 0.406], dtype=np.float32)
ECART = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def recadrer_feuille(img_bgr, taille=224):
    """Crop to the leaf box, pad to a square with the background colour, resize."""
    H, W = img_bgr.shape[:2]
    (cx, cy, bw, bh), _ = boite_feuille(img_bgr)
    x0, y0 = int(max(0, (cx - bw / 2) * W)), int(max(0, (cy - bh / 2) * H))
    x1, y1 = int(min(W, (cx + bw / 2) * W)), int(min(H, (cy + bh / 2) * H))
    crop = img_bgr[y0:y1, x0:x1] if (x1 - x0) > 8 and (y1 - y0) > 8 else img_bgr
    h, w = crop.shape[:2]
    cote = max(h, w)
    fond = [int(v) for v in np.median(img_bgr.reshape(-1, 3), axis=0)]
    carre = cv2.copyMakeBorder(crop, (cote - h) // 2, cote - h - (cote - h) // 2,
                               (cote - w) // 2, cote - w - (cote - w) // 2,
                               cv2.BORDER_CONSTANT, value=fond)
    return cv2.resize(carre, (taille, taille), interpolation=cv2.INTER_AREA)


def charger_modele(device, pretrained=True):
    import torch
    from torchvision import models
    poids = models.ResNet18_Weights.DEFAULT if pretrained else None
    modele = models.resnet18(weights=poids)
    modele.fc = torch.nn.Identity()
    return modele.eval().to(device)


def calculer_embeddings(chemins, batch=32, device=None, pretrained=True):
    """(n, 512) float32, L2-normalised, invariant to the 8 rotations/flips."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modele = charger_modele(device, pretrained)
    sorties = []
    with torch.no_grad():
        for debut in range(0, len(chemins), batch):
            lot = []
            for chemin in chemins[debut:debut + batch]:
                img = cv2.imread(str(chemin))
                if img is None:
                    raise ValueError(f"Unreadable image: {chemin}")
                rgb = cv2.cvtColor(recadrer_feuille(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                lot.append(((rgb - MOYENNE) / ECART).transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(lot)).to(device)                     # (B, 3, 224, 224)
            variantes = [torch.rot90(v, k, dims=(2, 3)) for v in (x, torch.flip(x, dims=(3,)))
                         for k in range(4)]
            feats = modele(torch.cat(variantes)).reshape(8, len(lot), -1).mean(0)
            sorties.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
            if (debut // batch) % 20 == 0:
                print(f"  embeddings {min(debut + batch, len(chemins))}/{len(chemins)}")
    return np.concatenate(sorties).astype(np.float32)


def voisins_plus_proches(E, classes):
    """For each image: similarity of its nearest neighbour in the same class / another class."""
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    classes = np.asarray(classes)
    meme = classes[:, None] == classes[None, :]
    nn_meme = np.where(meme, S, -1.0).max(1)
    nn_autre = np.where(~meme, S, -1.0).max(1)
    return S, nn_meme, nn_autre


def calibrer_seuil(nn_autre, quantile=0.995, seuil_min=0.90):
    """Same-leaf threshold = high quantile of 'nearest different-class leaf' similarity."""
    return float(max(seuil_min, np.quantile(nn_autre, quantile)))


def grouper_par_similarite(S, seuil):
    """Union-find over pairs with similarity >= seuil. Returns a group id per image."""
    n = len(S)
    parent = list(range(n))

    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(S >= seuil, k=1))):
        ri, rj = racine(int(i)), racine(int(j))
        if ri != rj:
            parent[rj] = ri
    ids = defaultdict(lambda: len(ids))
    return [ids[racine(i)] for i in range(n)]
