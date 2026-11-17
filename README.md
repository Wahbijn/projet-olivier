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
│   ├── raw/                  # images brutes (non versionnées, voir « Données »)
│   └── preprocessing/        # splits.csv : découpage train / val / test / externe
├── notebooks/
│   └── analysis.ipynb        # EDA, nettoyage, entraînement, évaluation, drift, Grad-CAM
├── src/
│   ├── data/                 # preprocessing.py
│   ├── features/             # feature_extraction.py
│   └── models/               # train.py, test.py
├── models/                   # model_V1.pt, model_V1.onnx (non versionnés dans Git)
├── Resultats/                # matrices de confusion, Grad-CAM, drift, tableaux CSV
├── configs/                  # config.yaml, param.yaml
├── tests/                    # tests automatiques
├── mlflow.db                 # suivi des expériences MLflow
├── requirements.txt
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

---

## ⚙️ Installation

Version recommandée : **Python 3.10**.

```bash
git clone https://github.com/TON_PSEUDO/projet-olivier.git
cd projet-olivier

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

---

## 🚀 Utilisation

**Vérifier le modèle exporté** (chargement du `.pt` et cohérence avec l'ONNX) :

```bash
python tests/verifier_modele.py
```

**Consulter les expériences MLflow :**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrir ensuite http://127.0.0.1:5000 dans le navigateur.

**Reproduire l'entraînement :** ouvrir `notebooks/analysis.ipynb` sur Kaggle (GPU T4), ajouter les deux jeux
de données en entrée, puis lancer *Run All*.

---

## 🛣️ Feuille de route

- [x] EDA, nettoyage et détection des doublons
- [x] Baseline ResNet50 + modèle avancé EfficientNet-B0
- [x] Évaluation, test de drift (Kolmogorov-Smirnov), Grad-CAM
- [x] Suivi MLflow, export ONNX
- [ ] Versionnement des données et des modèles avec DVC
- [ ] API FastAPI (`/health`, `/predict`)
- [ ] Interface web (React)
- [ ] Docker et docker-compose
- [ ] CI/CD : GitHub Actions + pytest
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
