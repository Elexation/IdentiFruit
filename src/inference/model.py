"""
model.py — Builds, saves, and loads the DINOv2 ViT-B/14 model.
"""

import os
import torch
import torch.nn as nn


DINOV2_EMBED_DIM = 768  # ViT-B output vector constant.


class DINOv2Classifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone with a linear classification head.

    backbone: pretrained DINOv2 — outputs (B, 768) CLS token per image
    head:     Linear(768, num_classes) — our task-specific output layer
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(DINOV2_EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone returns the CLS token: (B, 768)
        # head maps it to class logits: (B, num_classes)
        return self.head(self.backbone(x))


def build_model(num_classes: int) -> nn.Module:
    """
    Builds a DINOv2 ViT-B/14 model adapted for our fruit classes.

    Args:
        num_classes: The output classes needed

    Returns:
        The model ready for training or inference.
    """
    
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    return DINOv2Classifier(backbone, num_classes)


def save_model(model: nn.Module, class_names: list, path: str) -> None:
    """
    Saves the model weights and class name list to a .pt file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "classes":     class_names,
    }, path)

    print(f"  Saved -> {path}")


def load_model(path: str, device: torch.device):
    """
    Loads a saved model and its class names from a .pt file.

    Args:
        path:   File path to load from (e.g., "models/fruit_model.pt")
        device: Where to put the model (CPU or CUDA GPU)

    Returns:
        (model, class_names) tuple — model is ready for inference
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    class_names = checkpoint["classes"]

    model = build_model(len(class_names)) # Rebuild the model architecture with the correct number of classes

    model.load_state_dict(checkpoint["model_state"]) # Load the saved weights into the model
    model.to(device)

    return model, class_names
