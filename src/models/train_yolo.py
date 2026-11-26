"""
DVC stage `entrainement_yolo`: train the YOLO leaf-disease detector and evaluate it on the test split.

Usage:
    python -m src.models.train_yolo

Inputs:  data/processed/yolo/, configs/param.yaml (detection.yolo)
Outputs:
    models/yolo_best.pt                   best weights (DVC out -> `dvc push`)
    reports/metrics_yolo.json             test metrics (DVC metric)
    reports/yolo/                         training curves, confusion matrix, PR curves
    runs/                                 Ultralytics working folder (git-ignored, not versioned)

Metrics logged to MLflow (MLFLOW_TRACKING_URI from .env -> DagsHub), with the tag author=TEAM_MEMBER.
Besides mAP, we compute an image-level accuracy / macro-F1 (top box class vs true class)
so the detector can be compared fairly with the EfficientNet classifier on the same test split.
"""
import json
import os
import shutil
from pathlib import Path

import mlflow
import yaml
from sklearn.metrics import accuracy_score, f1_score

RACINE = Path(__file__).resolve().parents[2]
DONNEES = RACINE / "data" / "processed" / "yolo"
MODELE_SORTIE = RACINE / "models" / "yolo_best.pt"
METRIQUES = RACINE / "reports" / "metrics_yolo.json"
FIGURES = RACINE / "reports" / "yolo"
RUNS = RACINE / "runs"

try:
    from dotenv import load_dotenv
    load_dotenv(RACINE / ".env")
except ImportError:
    pass


def charger_yaml(chemin):
    with open(RACINE / chemin, encoding="utf-8") as f:
        return yaml.safe_load(f)


def yaml_absolu():
    """Copy data.yaml with an absolute `path` (Ultralytics needs it; never committed)."""
    donnees = yaml.safe_load((DONNEES / "data.yaml").read_text(encoding="utf-8"))
    donnees["path"] = DONNEES.as_posix()
    RUNS.mkdir(exist_ok=True)
    cible = RUNS / "data_runtime.yaml"
    cible.write_text(yaml.safe_dump(donnees, allow_unicode=True), encoding="utf-8")
    return cible, donnees["names"]


def precision_image(modele, noms, imgsz, conf):
    """Image-level accuracy / macro-F1: class of the most confident box ('none' if no box)."""
    vrais, preds = [], []
    images = sorted((DONNEES / "images" / "test").glob("*.jpg"))
    for res in modele.predict([str(p) for p in images], imgsz=imgsz, conf=conf, stream=True, verbose=False):
        etiquette = (DONNEES / "labels" / "test" / (Path(res.path).stem + ".txt")).read_text().split()[0]
        vrais.append(int(etiquette))
        preds.append(int(res.boxes.cls[res.boxes.conf.argmax()]) if len(res.boxes) else -1)
    return accuracy_score(vrais, preds), f1_score(vrais, preds, average="macro", labels=list(noms))


def main():
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})          # we log ourselves (run name, tags, artefacts)

    p = charger_yaml("configs/param.yaml")["detection"]["yolo"]
    data_yaml, noms = yaml_absolu()
    membre = os.getenv("TEAM_MEMBER", "unknown")

    mlflow.set_experiment(p["experience"])
    with mlflow.start_run(run_name=f"{membre}-{Path(p['modele']).stem}-imgsz{p['imgsz']}"):
        mlflow.set_tags({"author": membre, "task": "detection", "pipeline": "dvc"})
        mlflow.log_params(p)

        modele = YOLO(p["modele"])
        modele.train(data=str(data_yaml), imgsz=p["imgsz"], epochs=p["epochs"], batch=p["batch"],
                     patience=p["patience"], seed=p["seed"], deterministic=True, workers=p["workers"],
                     project=str(RUNS / "detect"), name="olive", exist_ok=True, verbose=True)
        dossier_run = Path(modele.trainer.save_dir)
        meilleur = dossier_run / "weights" / "best.pt"

        modele = YOLO(str(meilleur))
        m = modele.val(data=str(data_yaml), split="test", imgsz=p["imgsz"],
                       project=str(RUNS / "detect"), name="olive_test", exist_ok=True)
        acc, f1 = precision_image(modele, noms, p["imgsz"], p["conf_image"])
        metriques = {
            "test_map50": float(m.box.map50), "test_map50_95": float(m.box.map),
            "test_precision": float(m.box.mp), "test_recall": float(m.box.mr),
            "test_image_accuracy": float(acc), "test_image_f1_macro": float(f1),
            **{f"test_map50_95_{noms[i]}": float(v) for i, v in enumerate(m.box.maps)},
        }
        mlflow.log_metrics(metriques)

        MODELE_SORTIE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meilleur, MODELE_SORTIE)
        METRIQUES.parent.mkdir(parents=True, exist_ok=True)
        METRIQUES.write_text(json.dumps(metriques, indent=2), encoding="utf-8")
        if FIGURES.exists():
            shutil.rmtree(FIGURES)
        FIGURES.mkdir(parents=True)
        for f in list(dossier_run.glob("*.png")) + [dossier_run / "results.csv"]:
            if f.exists():
                shutil.copy(f, FIGURES / f.name)

        mlflow.log_artifact(str(METRIQUES))
        mlflow.log_artifacts(str(FIGURES), artifact_path="figures")
        mlflow.log_artifact(str(MODELE_SORTIE), artifact_path="model")
        print(json.dumps(metriques, indent=2))


if __name__ == "__main__":
    main()
