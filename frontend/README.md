# Zitouna – interface web

Interface React + Vite + TypeScript pour l'API de détection des maladies de l'olivier
(healthy, aculus, peacock_spot). Mobile-first, bilingue FR / AR (RTL).

## Installation

```bash
cd frontend
npm install
```

## Lancement en développement

```bash
npm run dev          # http://localhost:5173
```

L'API doit tourner en parallèle (depuis la racine du projet) :

```bash
uvicorn api.main:app --reload    # http://127.0.0.1:8000
```

## Variable d'environnement `VITE_API_URL`

URL de base de l'API (défaut : `http://127.0.0.1:8000`). Pour la changer, créer `frontend/.env.local` :

```
VITE_API_URL=http://192.168.1.20:8000
```

Utile pour tester depuis un smartphone sur le même réseau (lancer aussi uvicorn avec `--host 0.0.0.0`).
La valeur est figée **au moment du build** (Vite).

## Build

```bash
npm run build        # vérifie les types puis génère dist/
npm run preview      # sert dist/ en local
```

## Docker

```bash
docker build -t zitouna-frontend --build-arg VITE_API_URL=http://127.0.0.1:8000 .
docker run -p 8080:80 zitouna-frontend
```

## Structure

- `src/components/` : UploadZone, ResultCard, ConfidenceGauge, HeatmapCompare, DiseaseInfo, History, ApiStatus, Ornaments (SVG)
- `src/data/maladies.ts` : fiches maladies (FR/AR), textes indicatifs et prudents
- `src/api.ts` : appels `/health` et `/predict`, gestion des erreurs 400 / 413 / 503 / API injoignable
- `src/i18n.tsx` : traductions et direction RTL
- `src/prepareImage.ts` : reproduit côté client le prétraitement du modèle (padding carré + 224×224) pour le comparateur
