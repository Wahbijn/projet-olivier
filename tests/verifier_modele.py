# verifier.py : vérifie que les modèles fonctionnent hors de Kaggle
import numpy as np, torch, torch.nn as nn, onnxruntime as ort
from torchvision import models

# 1) Le .pt se charge-t-il, avec ses informations ?
ckpt = torch.load("models/model_V1.pt", map_location="cpu", weights_only=False)
print("Architecture :", ckpt["architecture"])
print("Classes      :", ckpt["classes"])
print("Seuil        :", ckpt["seuil_confiance"])
print("Métriques    :", ckpt["metriques"])

m = models.resnet50()
m.fc = nn.Linear(m.fc.in_features, 3)
m.load_state_dict(ckpt["state_dict"])
m.eval()

# 2) Le .onnx donne-t-il les mêmes résultats que le .pt ?
x = np.random.rand(2, 3, 224, 224).astype(np.float32)
with torch.no_grad():
    sortie_pt = m(torch.from_numpy(x)).numpy()
session = ort.InferenceSession("models/model_V1.onnx")
sortie_onnx = session.run(None, {"image": x})[0]

ecart = np.abs(sortie_pt - sortie_onnx).max()
print("\nÉcart max PyTorch vs ONNX :", ecart)
print("✅ OK, les deux modèles sont identiques" if ecart < 1e-3 else "❌ Problème : les sorties diffèrent")