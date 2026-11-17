"""Évaluation des modèles entraînés + export du modèle déployé (notebook, sections 7 à 9).

Lancement (après src/models/train.py) :
    python -m src.models.test

Pour chaque modèle de configs/param.yaml présent dans models/ :
    - test Uğuz (3 classes) : exactitude, F1 macro, matrice de confusion
    - test externe Zeytin (binaire sain / malade) : exactitude, F1, faux malades, malades manqués
    - coût : paramètres, taille, temps CPU par image
Pour le modèle « deploye: true » :
    - tableau des seuils de confiance, test KS (drift)
    - export models/model_V1.pt + models/model_V1.onnx (vérifié) et Resultats/metriques.json
"""
import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from src.features.feature_extraction import transformations
from src.models.architectures import creer_modele
from src.models.train import OliveDataset, charger_donnees, predire
from src.utils import RACINE, charger_config, charger_params

SEUILS = [0.0, 0.6, 0.7, 0.8, 0.9]


def binaire(classes):
    return (np.asarray(classes) != 0).astype(int)        # 0 = sain, 1 = malade


def charger_modele(chemin, appareil):
    ckpt = torch.load(chemin, map_location="cpu", weights_only=False)
    m = creer_modele(ckpt["architecture"], len(ckpt["classes"]), pretrained=False)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(appareil).eval(), ckpt


def temps_cpu_ms(modele, repetitions=10):
    m = copy.deepcopy(modele).cpu().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3):
            m(x)
        debut = time.time()
        for _ in range(repetitions):
            m(x)
    return (time.time() - debut) / repetitions * 1000


def exporter_onnx(modele, chemin):
    m = copy.deepcopy(modele).cpu().eval()
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(m, x, str(chemin), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    import onnxruntime as ort
    lot = np.random.rand(2, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(lot)).numpy()
    sortie = ort.InferenceSession(str(chemin)).run(None, {"image": lot})[0]
    ecart = float(np.abs(ref - sortie).max())
    if ecart > 1e-3:
        raise RuntimeError(f"Export ONNX incohérent (écart max {ecart:.2e})")
    return ecart


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seuil", type=float, default=None,
                    help="seuil de confiance du modèle déployé (défaut : configs/config.yaml)")
    args = ap.parse_args()

    config, params = charger_config(), charger_params()
    classes = config["classes"]
    seuil = args.seuil if args.seuil is not None else config["modele_deploye"]["seuil_confiance"]
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    dossier_res = RACINE / config["resultats"]
    dossier_res.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config["suivi"]["mlflow_uri"])

    X, y, idx, _ = charger_donnees(classes)
    _, transfo_eval = transformations(config["pretraitement"]["taille_image"])
    nw = params["entrainement"].get("num_workers", 2)
    loaders = {s: DataLoader(OliveDataset(y, idx[s], transfo_eval), batch_size=64, num_workers=nw)
               for s in ["test", "externe"]}

    disponibles = [n for n in params["modeles"] if (RACINE / "models" / f"{n}.pt").exists()]
    if not disponibles:
        raise SystemExit("Aucun modèle dans models/ : lancer d'abord src/models/train.py")
    deploye = next((n for n, p in params["modeles"].items() if p.get("deploye")), None)

    comparaison, bilan, sorties = [], [], {}
    fig, axes = plt.subplots(1, len(disponibles), figsize=(5 * len(disponibles), 4.5), squeeze=False)
    for ax, nom in zip(axes[0], disponibles):
        chemin = RACINE / "models" / f"{nom}.pt"
        modele, ckpt = charger_modele(chemin, appareil)
        yt, yp, pr = predire(modele, loaders["test"], appareil)
        ye, ype, pre = predire(modele, loaders["externe"], appareil)
        sorties[nom] = (modele, ckpt, yt, yp, pr, ye, ype, pre)

        print("=" * 60, f"\n{nom}\n", classification_report(yt, yp, target_names=classes, digits=3))
        ConfusionMatrixDisplay(confusion_matrix(yt, yp, labels=range(len(classes))),
                               display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(nom, fontsize=10)

        mc = confusion_matrix(binaire(ye), binaire(ype), labels=[0, 1])
        metriques = {
            "f1_macro_test_uguz": f1_score(yt, yp, average="macro"),
            "acc_test_uguz": accuracy_score(yt, yp),
            "acc_binaire_zeytin": accuracy_score(binaire(ye), binaire(ype)),
            "f1_binaire_zeytin": f1_score(binaire(ye), binaire(ype)),
            "faux_malades_zeytin": int(mc[0, 1]),
            "malades_manques_zeytin": int(mc[1, 0]),
            "confiance_moy_zeytin": float(pre.max(1).mean()),
            "parametres_M": sum(p.numel() for p in modele.parameters()) / 1e6,
            "taille_Mo": chemin.stat().st_size / 1e6,
            "temps_cpu_ms": temps_cpu_ms(modele),
        }
        comparaison.append({"modèle": nom, "exactitude": metriques["acc_test_uguz"],
                            "F1 macro": metriques["f1_macro_test_uguz"],
                            "paramètres (M)": metriques["parametres_M"],
                            "taille fichier (Mo)": metriques["taille_Mo"],
                            "temps CPU (ms/img)": metriques["temps_cpu_ms"]})
        bilan.append({"modèle": nom, "F1 macro test Uğuz (3 classes)": metriques["f1_macro_test_uguz"],
                      "acc binaire Zeytin": metriques["acc_binaire_zeytin"],
                      "F1 binaire Zeytin": metriques["f1_binaire_zeytin"],
                      f"faux malades (/{int(mc[0].sum())})": metriques["faux_malades_zeytin"],
                      f"malades manqués (/{int(mc[1].sum())})": metriques["malades_manques_zeytin"],
                      "confiance moy. Zeytin": metriques["confiance_moy_zeytin"]})
        sorties[nom] += (metriques,)

        # les métriques finales rejoignent le run MLflow de l'entraînement
        if ckpt.get("mlflow_run_id"):
            with mlflow.start_run(run_id=ckpt["mlflow_run_id"]):
                mlflow.log_metrics(metriques)

    plt.suptitle("Matrices de confusion — test Uğuz")
    plt.tight_layout()
    plt.savefig(dossier_res / "matrices_confusion_test.png", dpi=150)
    plt.close()
    pd.DataFrame(comparaison).round(3).to_csv(dossier_res / "comparaison_modeles.csv", index=False)
    pd.DataFrame(bilan).round(3).to_csv(dossier_res / "bilan_drift.csv", index=False)
    print(pd.DataFrame(bilan).round(3).to_string(index=False))

    if deploye not in sorties:
        print(f"\n⚠️ Modèle déployé « {deploye} » absent de models/ : pas d'export.")
        return

    # ---------------- modèle déployé : seuils, drift, export
    modele, ckpt, yt, yp, pr, ye, ype, pre, metriques = sorties[deploye]
    conf_t, conf_e = pr.max(1), pre.max(1)
    erreurs_e = binaire(ye) != binaire(ype)
    lignes = []
    for s in SEUILS:
        gt, ge = conf_t >= s, conf_e >= s
        lignes.append({
            "seuil": s,
            "% test Uğuz rejeté": 100 * (1 - gt.mean()),
            "exactitude test Uğuz (images gardées)": accuracy_score(yt[gt], yp[gt]) if gt.any() else np.nan,
            "% Zeytin rejeté": 100 * (1 - ge.mean()),
            "exactitude Zeytin binaire (gardées)":
                accuracy_score(binaire(ye[ge]), binaire(ype[ge])) if ge.any() else np.nan,
            "% des erreurs Zeytin interceptées":
                100 * (erreurs_e & ~ge).sum() / max(erreurs_e.sum(), 1),
        })
    pd.DataFrame(lignes).round(3).to_csv(dossier_res / "seuils_confiance.csv", index=False)

    lum_t = X[idx["test"]].mean(axis=(1, 2, 3))
    lum_e = X[idx["externe"]].mean(axis=(1, 2, 3))
    ks_lum, ks_conf = ks_2samp(lum_t, lum_e), ks_2samp(conf_t, conf_e)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (a, b), titre in [(axes[0], (lum_t, lum_e), f"Luminosité moyenne (KS D={ks_lum.statistic:.2f})"),
                              (axes[1], (conf_t, conf_e), f"Confiance {deploye} (KS D={ks_conf.statistic:.2f})")]:
        ax.hist(a, bins=30, alpha=0.6, density=True, label="test Uğuz")
        ax.hist(b, bins=30, alpha=0.6, density=True, label="Zeytin")
        ax.set_title(titre)
        ax.legend()
    plt.suptitle("Détection de drift : Uğuz vs Zeytin")
    plt.tight_layout()
    plt.savefig(dossier_res / "drift_ks.png", dpi=150)
    plt.close()

    garde_t = conf_t >= seuil
    final = {
        **{k: round(float(v), 4) for k, v in metriques.items()},
        "seuil_confiance": seuil,
        "rejet_test_uguz_pct": round(100 * (1 - garde_t.mean()), 2),
        "ks_luminosite_D": round(float(ks_lum.statistic), 4),
        "ks_confiance_D": round(float(ks_conf.statistic), 4),
        "modele": deploye,
    }
    (dossier_res / "metriques.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    # Checkpoint de production : même format que le notebook (lu par predict.py)
    ckpt_prod = {k: ckpt[k] for k in ["state_dict", "architecture", "classes", "taille_entree", "moyenne", "ecart"]}
    ckpt_prod.update({
        "seuil_confiance": seuil,
        "metriques": {"f1_macro_test_uguz": metriques["f1_macro_test_uguz"],
                      "acc_binaire_zeytin": metriques["acc_binaire_zeytin"]},
        "source": deploye,
        "mlflow_run_id": ckpt.get("mlflow_run_id"),
    })
    torch.save(ckpt_prod, RACINE / config["modele_deploye"]["pytorch"])
    ecart = exporter_onnx(modele, RACINE / config["modele_deploye"]["onnx"])
    if ckpt.get("mlflow_run_id"):
        with mlflow.start_run(run_id=ckpt["mlflow_run_id"]):
            mlflow.set_tag("deploye", "oui")
            mlflow.log_param("seuil_confiance", seuil)
            for f in ["matrices_confusion_test.png", "drift_ks.png", "seuils_confiance.csv", "metriques.json"]:
                mlflow.log_artifact(str(dossier_res / f))
    print(f"\n✅ {deploye} exporté -> models/model_V1.pt + model_V1.onnx (écart ONNX {ecart:.1e})")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
