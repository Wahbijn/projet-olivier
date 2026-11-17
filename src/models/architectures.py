"""Création des modèles (même tête de classification que dans le notebook,
pour rester compatible avec models/model_V1.pt déjà entraîné)."""
import torch.nn as nn
from torchvision import models


def creer_modele(architecture: str, nb_classes: int = 3, pretrained: bool = True) -> nn.Module:
    if architecture == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, nb_classes)
    elif architecture == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nb_classes)
    else:
        raise ValueError(f"Architecture inconnue : {architecture}")
    return m


def couche_gradcam(modele: nn.Module, architecture: str) -> nn.Module:
    """Dernière couche convolutive, utilisée par Grad-CAM."""
    if architecture == "resnet50":
        return modele.layer4[-1]
    if architecture == "efficientnet_b0":
        return modele.features[-1]
    raise ValueError(f"Architecture inconnue : {architecture}")
