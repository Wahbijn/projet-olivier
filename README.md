# 🫒 Détection de maladies de l'olivier par Deep Learning

Projet de 3e année Ingénierie IA & Data Science, pour les cours **Deep Learning Avancé** et **MLOps**.

À partir d'une **photo de feuille d'olivier**, le modèle prédit son état, donne un **score de confiance** et une
**heatmap Grad-CAM** qui montre les zones ayant guidé sa décision. Si la confiance est trop faible,
il répond **« incertain »** au lieu de risquer un mauvais diagnostic.

| Classe | Description |
| :--- | :--- |
| `healthy` | Feuille saine |
| `aculus` | Attaque de l'acarien *Aculus olearius* |
| `peacock_spot` | Œil de paon (*Spilocaea oleagina*) |

---

## 📊 Résultats principaux

| Modèle | F1 macro (test Uğuz) | Exactitude binaire Zeytin (drift) | Paramètres | Temps CPU / image |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet50 (déployé)** | 0,974 | **0,922** | 23,5 M | 81 ms |
| EfficientNet-B0 | **0,985** | 0,795 | 4,0 M | 34 ms |
| EfficientNet-B0 (augmentation renforcée) | 0,969 | 0,894 | 4,0 M | 34 ms |

- **Modèle retenu : ResNet50**, le plus robuste sur des images venant d'une autre source.
- **Seuil de confiance : 0,8.** Il rejette 4,8 % des images Uğuz et intercepte 69 % des erreurs sur Zeytin.
- **Qualité des données :** 107 groupes de doublons fuyaient entre le train et le test d'origine d'Uğuz.
  Ils ont été supprimés avant notre propre découpage.

Le détail complet est dans [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

---

## 📁 Structure du projet

```
projet-olivier/
├── data/
│   ├── raw/                  # images brutes : uguz/ et zeytin/ (DVC, pas Git — voir « Données »)
│   ├── preprocessing/        # splits.csv : découpage de référence train / val / test / externe
│   └── processed/            # images_224.npy, généré par le pipeline (DVC)
├── notebooks/
│   └── analysis.ipynb        # EDA, nettoyage, entraînement, évaluation, drift, Grad-CAM
├── src/
│   ├── data/preprocessing.py         # vérification / régénération du découpage (dédoublonnage dHash)
│   ├── features/feature_extraction.py# prétraitement des images (partagé entraînement + API)
│   ├── models/architectures.py       # ResNet50 / EfficientNet-B0
│   ├── models/train.py               # entraînement d'un modèle de param.yaml + suivi MLflow
│   ├── models/evaluate.py            # évaluation, drift, seuil de confiance, export .pt / .onnx
│   ├── models/predict.py             # prédiction + Grad-CAM (utilisé par l'API)
│   └── utils.py
├── api/                      # API FastAPI (/health, /predict) + Dockerfile
├── frontend/                 # interface web React (Vite) + Dockerfile
├── models/                   # *.pt, model_V1.onnx (DVC, pas Git)
├── Resultats/                # matrices de confusion, Grad-CAM, drift, tableaux CSV, metriques.json
├── configs/                  # config.yaml (chemins, classes, déploiement), param.yaml (hyperparamètres)
├── tests/                    # pytest : prétraitement, modèle exporté, API
├── dvc.yaml                  # pipeline reproductible : preparation → entrainement → evaluation
├── docker-compose.yml        # API + interface
├── .github/workflows/ci.yml  # tests + build interface + build image API à chaque push / PR
├── mlflow.db                 # suivi des expériences MLflow
├── requirements.txt          # environnement complet (entraînement, API, tests)
├── requirements-api.txt      # dépendances minimales de l'image Docker de l'API
└── README.md
```

---

## 🗂️ Données

Deux jeux publics Kaggle (à télécharger, ils ne sont pas inclus dans le dépôt) :

| Jeu | Images | Rôle | Lien |
| :--- | :---: | :--- | :--- |
| Olive Leaf Image Dataset (Uğuz & Uysal, 2021) | 3 400 | Entraînement / validation / test | [Kaggle](https://www.kaggle.com/datasets/habibulbasher01644/olive-leaf-image-dataset) |
| Zeytin | 954 | Test externe (drift), évalué en sain / malade | [Kaggle](https://www.kaggle.com/datasets/serhathoca/zeytin) |

Après dédoublonnage, il reste **4 120 images** : 2 216 pour le train, 475 pour la validation,
475 pour le test et 954 en externe.

Après téléchargement, placer les deux jeux ainsi (les chemins de `splits.csv` sont relatifs à `data/raw/`) :

```
data/raw/
├── uguz/dataset/{train,test}/{Healthy,aculus_olearius,olive_peacock_spot}/...
└── zeytin/{sağlam,hastalıklı}/...
```

Puis vérifier que tout est en place : `python -m src.data.preprocessing`

> Les images Zeytin « hastalıklı » (malades) sont étiquetées `malade` : ce jeu sert uniquement
> d'évaluation binaire sain / malade (voir notebook, section 8).

---

## ⚙️ Installation

Version recommandée : **Python 3.10** (3.11 fonctionne aussi).

```bash
git clone https://github.com/Wahbijn/projet-olivier.git
cd projet-olivier

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# avec un GPU NVIDIA : installer d'abord torch/torchvision depuis https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Pipeline complet (DVC)

```bash
dvc repro                         # préparation → entraînement des 4 modèles → évaluation + export
dvc repro entrainement@resnet50_baseline   # un seul modèle
dvc metrics show                  # métriques du modèle déployé (Resultats/metriques.json)
dvc params diff                   # hyperparamètres modifiés depuis le dernier commit
```

DVC ne relance que les étapes dont une dépendance (code, données, `configs/param.yaml`) a changé.
Les étapes peuvent aussi être lancées une par une :

```bash
python -m src.features.feature_extraction              # images 224x224 -> data/processed/images_224.npy
python -m src.models.train --modele resnet50_baseline  # un modèle de configs/param.yaml
python -m src.models.evaluate                          # comparaison, drift, seuils, export model_V1.pt / .onnx
```

Le découpage `data/preprocessing/splits.csv` est la référence des résultats ci-dessus ; il ne se régénère
qu'explicitement (`python -m src.data.preprocessing --regenerer`).

### API et interface

```bash
uvicorn api.main:app --reload       # API : http://127.0.0.1:8000/docs
cd frontend && npm install && npm run dev   # interface (voir frontend/README.md)
```

Ou tout en Docker (nécessite `models/model_V1.pt`) :

```bash
docker compose up --build           # API : http://localhost:8000  |  interface : http://localhost:8080
```

### Tests

```bash
pytest -v       # les tests qui ont besoin de models/model_V1.pt sont ignorés s'il est absent
```

**Consulter les expériences MLflow :**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrir ensuite http://127.0.0.1:5000 dans le navigateur. Deux expériences :

- `olive-disease` : les 4 runs historiques du notebook Kaggle (métriques et courbes) ;
- `olive-disease-pipeline` : les runs lancés par `src/models/train.py`, avec la version du découpage
  (`version_splits`), le commit Git et les artefacts (matrices de confusion, drift, seuils).

Le notebook `notebooks/analysis.ipynb` reste la référence de l'analyse (EDA, Grad-CAM, interprétation) ;
il peut toujours être relancé sur Kaggle (GPU T4) avec les deux jeux de données en entrée.

---

## 🛣️ Feuille de route

- [x] EDA, nettoyage et détection des doublons
- [x] Baseline ResNet50 + modèle avancé EfficientNet-B0
- [x] Évaluation, test de drift (Kolmogorov-Smirnov), Grad-CAM
- [x] Suivi MLflow, export ONNX
- [x] Pipeline reproductible en scripts (`src/`) + DVC (`dvc.yaml`)
- [ ] Stockage distant DVC (remote partagé pour les données et les modèles)
- [x] API FastAPI (`/health`, `/predict`)
- [x] Interface web (React)
- [x] Docker et docker-compose
- [x] CI : GitHub Actions (pytest, build interface, build image API)
- [ ] Surveillance du drift en production (Evidently AI)
- [ ] Test sur de vraies photos de feuilles d'oliviers tunisiens

---

## 👥 Équipe

| Membre | Rôle |
| :--- | :--- |
| [Nom 1] | Modèles (Deep Learning) |
| [Nom 2] | Données et pipeline |
| [Nom 3] | Déploiement (API, interface, Docker) |
| [Nom 4] | Qualité, CI/CD, rapport |

---

## ⚠️ Avertissement

Cet outil est un **prototype pédagogique d'aide au diagnostic**. Il a été entraîné sur des photos prises en
conditions contrôlées et ne remplace pas l'avis d'un ingénieur agronome.
