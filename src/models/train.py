"""Entraînement d'un modèle défini dans configs/param.yaml (notebook, section 6).

Lancement (depuis la racine du projet) :
    python -m src.models.train --modele resnet50_baseline
    python -m src.models.train --modele efficientnet_b0_augfort

Sorties :
    models/<modele>.pt                         poids + métadonnées (format lu par predict.py)
    Resultats/historique_<modele>.csv          courbes par époque
    Run MLflow (expérience « olive-disease-pipeline ») avec paramètres, courbes, version des données
"""
import argparse
import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from src.features.feature_extraction import ECART, MOYENNE, chemin_images_preparees, transformations
from src.models.architectures import creer_modele
from src.utils import RACINE, charger_config, charger_params, commit_git, empreinte_fichier, fixer_graine


class OliveDataset(Dataset):
    """Images préparées (224x224, uint8) + étiquettes, lues dans images_224.npy.

    Le fichier est ouvert en mmap dans chaque worker (sous Windows, passer le tableau
    lui-même aux workers le recopierait entièrement en mémoire)."""

    def __init__(self, y, indices, transfo):
        self.y, self.indices, self.transfo = y, np.asarray(indices), transfo
        self.X = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        if self.X is None:
            self.X = np.load(chemin_images_preparees(), mmap_mode="r")
        i = self.indices[k]
        return self.transfo(np.array(self.X[i])), int(self.y[i])


def charger_donnees(classes):
    """Images préparées + splits.csv. Zeytin (externe) : 0 = sain, 1 = malade."""
    config = charger_config()
    splits = pd.read_csv(RACINE / config["donnees"]["splits"])
    X = np.load(chemin_images_preparees(), mmap_mode="r")
    assert len(X) == len(splits), "images_224.npy ne correspond pas à splits.csv : relancer la préparation"
    index = {c: i for i, c in enumerate(classes)}
    y = splits["classe"].map(lambda c: index.get(c, 1)).to_numpy()   # "malade" -> 1 (binaire)
    idx = {s: splits.index[splits["split"] == s].to_numpy() for s in ["train", "val", "test", "externe"]}
    return X, y, idx, splits


@torch.no_grad()
def predire(modele, loader, appareil):
    modele.eval()
    vrais, probas = [], []
    for images, etiquettes in loader:
        with torch.autocast(device_type="cuda", enabled=(appareil == "cuda")):
            sorties = modele(images.to(appareil))
        probas.append(torch.softmax(sorties.float(), dim=1).cpu())
        vrais.append(etiquettes)
    probas = torch.cat(probas).numpy()
    return torch.cat(vrais).numpy(), probas.argmax(1), probas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modele", required=True, help="clé de configs/param.yaml > modeles")
    ap.add_argument("--sans-preentrainement", action="store_true",
                    help="poids aléatoires (tests rapides uniquement)")
    args = ap.parse_args()

    config, params = charger_config(), charger_params()
    if args.modele not in params["modeles"]:
        raise SystemExit(f"Modèle inconnu. Choix : {list(params['modeles'])}")
    p = params["modeles"][args.modele]
    ent = params["entrainement"]
    classes = config["classes"]
    taille = config["pretraitement"]["taille_image"]
    fixer_graine(params["decoupage"]["seed"])

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {appareil}" + (f" ({torch.cuda.get_device_name(0)})" if appareil == "cuda" else ""))

    X, y, idx, _ = charger_donnees(classes)
    jitter = params["augmentation"][f"color_jitter_{p['color_jitter']}"]
    transfo_train, transfo_eval = transformations(taille, jitter)
    nw = ent.get("num_workers", 2)
    loaders = {
        "train": DataLoader(OliveDataset(y, idx["train"], transfo_train), batch_size=ent["batch_size"],
                            shuffle=True, num_workers=nw, persistent_workers=nw > 0),
        "val": DataLoader(OliveDataset(y, idx["val"], transfo_eval), batch_size=64,
                          num_workers=nw, persistent_workers=nw > 0),
    }
    print({s: len(v) for s, v in idx.items()})

    modele = creer_modele(p["architecture"], len(classes), pretrained=not args.sans_preentrainement).to(appareil)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=p["lr"])
    critere = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(appareil == "cuda"))

    dossier_modeles = RACINE / "models"
    dossier_modeles.mkdir(exist_ok=True)
    chemin_modele = dossier_modeles / f"{args.modele}.pt"
    dossier_res = RACINE / config["resultats"]
    dossier_res.mkdir(exist_ok=True)

    # MLflow : serveur défini par MLFLOW_TRACKING_URI (fichier .env), expérience du pipeline
    mlflow.set_experiment(config["suivi"]["experience_pipeline"])

    with mlflow.start_run(run_name=args.modele) as run:
        mlflow.set_tag("author", os.getenv("TEAM_MEMBER", "unknown"))
        mlflow.log_params({
            "modele": p["architecture"], "lr": p["lr"], "patience": p["patience"],
            "epochs_max": p["epochs_max"], "color_jitter": p["color_jitter"],
            "batch_size": ent["batch_size"], "optimiseur": ent["optimiseur"], "taille_entree": taille,
            "decoupage": "70/15/15 stratifie apres dedoublonnage", "seed": params["decoupage"]["seed"],
            "preentraine": not args.sans_preentrainement,
        })
        mlflow.set_tags({
            "version_splits": empreinte_fichier(RACINE / config["donnees"]["splits"])[:12],
            "commit_git": commit_git(),
            "appareil": appareil,
            "deploye": "candidat" if p.get("deploye") else "non",
        })

        meilleur_f1, attente, historique, debut_total = -1.0, 0, [], time.time()
        for epoch in range(1, p["epochs_max"] + 1):
            debut = time.time()
            modele.train()
            somme_perte, bons, total = 0.0, 0, 0
            for images, etiquettes in loaders["train"]:
                images, etiquettes = images.to(appareil), etiquettes.to(appareil)
                optimiseur.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=(appareil == "cuda")):
                    sorties = modele(images)
                    perte = critere(sorties, etiquettes)
                scaler.scale(perte).backward()
                scaler.step(optimiseur)
                scaler.update()
                somme_perte += perte.item() * len(etiquettes)
                bons += (sorties.argmax(1) == etiquettes).sum().item()
                total += len(etiquettes)

            y_val, p_val, _ = predire(modele, loaders["val"], appareil)
            ligne = {
                "epoch": epoch,
                "perte_train": somme_perte / total,
                "acc_train": bons / total,
                "acc_val": accuracy_score(y_val, p_val),
                "f1_val": f1_score(y_val, p_val, average="macro"),
            }
            historique.append(ligne)
            mlflow.log_metrics({k: v for k, v in ligne.items() if k != "epoch"}, step=epoch)
            print(f"Époque {epoch:2d} | perte {ligne['perte_train']:.3f} | acc train {ligne['acc_train']:.3f} "
                  f"| acc val {ligne['acc_val']:.3f} | F1 val {ligne['f1_val']:.3f} | {time.time() - debut:.0f}s")

            if ligne["f1_val"] > meilleur_f1:
                meilleur_f1, attente = ligne["f1_val"], 0
                torch.save({
                    "state_dict": modele.state_dict(),
                    "architecture": p["architecture"],
                    "classes": classes,
                    "taille_entree": taille,
                    "moyenne": MOYENNE,
                    "ecart": ECART,
                    "modele": args.modele,
                    "epoch": epoch,
                    "meilleur_f1_val": meilleur_f1,
                    "mlflow_run_id": run.info.run_id,
                }, chemin_modele)
                print("   ↳ meilleur modèle sauvegardé")
            else:
                attente += 1
                if attente >= p["patience"]:
                    print(f"Arrêt anticipé (pas d'amélioration depuis {p['patience']} époques)")
                    break

        chemin_hist = dossier_res / f"historique_{args.modele}.csv"
        pd.DataFrame(historique).to_csv(chemin_hist, index=False)
        mlflow.log_metrics({"meilleur_f1_val": meilleur_f1,
                            "duree_entrainement_min": (time.time() - debut_total) / 60})
        mlflow.log_artifact(str(chemin_hist))
        print(f"\nMeilleur F1 validation ({args.modele}) : {meilleur_f1:.4f} -> {chemin_modele.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
