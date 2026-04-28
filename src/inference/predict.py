"""
predict.py — Uses the trained model to classify a single PIL image as "fruit" and "freshness".
"""

import torch
from PIL import Image

from src.inference.model import load_model
from src.training.dataset import get_val_transform

MODEL_PATH = "models/fruit_model.pt"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.8

# Lazy-loaded on first predict() call
_model = None
_class_names = None
_device = None
_transform = None


def _load():
	"""Loads the model and transform once and caches them for all future calls."""
	global _model, _class_names, _device, _transform

	_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_model, _class_names = load_model(MODEL_PATH, _device)
	_model.eval()

	_transform = get_val_transform(IMAGE_SIZE)


def predict(img: Image.Image) -> dict:
	"""
	Classifies a single PIL image.

	Returns:
		{"fruit": str, "freshness": str|None, "confidence": float|None}
	"""
	if _model is None:
		_load()

	img = img.convert("RGB")

	tensor = _transform(img).unsqueeze(0).to(_device)

	with torch.no_grad():
		logits = _model(tensor)

	probs = torch.softmax(logits, dim=1)[0]
	pred_idx = probs.argmax().item()
	confidence = probs[pred_idx].item()
	pred_class = _class_names[pred_idx]

	if confidence < CONFIDENCE_THRESHOLD or pred_class == "unknown":
		return {"fruit": "Unknown", "freshness": None, "confidence": None}

	fruit, freshness = pred_class.rsplit("_", 1)
	return {"fruit": fruit.capitalize(), "freshness": freshness, "confidence": confidence}
