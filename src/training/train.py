"""
train.py — Two-phase DINOv2 training: freeze backbone then full fine-tune.
"""

import os
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.inference.model import build_model, save_model, load_model
from src.training.dataset import FruitDataset


DATA_DIR = "data/Train"
MODEL_PATH = "models/fruit_model.pt"
RESUME_PATH = "models/resume.pt"
IMAGE_SIZE = 224
BATCH_SIZE = 16
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
PHASE1_LR = 1e-3
PHASE2_LR = 5e-5
NUM_WORKERS = min(max(1, (os.cpu_count() or 2) // 2), 16)
USE_AMP = False
MAX_GRAD_NORM = 1.0

# Auto hardware detection
if torch.cuda.is_available():
	if torch.cuda.is_bf16_supported(): # BF16 is faster. see if supported.
		USE_AMP = True
	_vram = torch.cuda.get_device_properties(0).total_memory
	if _vram >= 20 * 1024**3:
		BATCH_SIZE = 256


# time formatting helper
def _fmt_time(secs):
	h = int(secs // 3600)
	m = int((secs % 3600) // 60)
	s = int(secs % 60)
	if h:
		return f"{h}h {m}m {s}s"
	if m:
		return f"{m}m {s}s"
	return f"{s}s"


def _save_resume(phase, epoch, model, optimizer, scheduler, best_val_loss, class_names):
	"""Saves a resume checkpoint"""
	torch.save({
		"phase": phase,
		"epoch": epoch,
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"scheduler_state": scheduler.state_dict() if scheduler else None,
		"best_val_loss": best_val_loss,
		"class_names": class_names,
	}, RESUME_PATH)


def train_one_epoch(model, loader, criterion, optimizer, device):
	"""Trains one epoch. Uses BF16 autocast if USE_AMP is enabled."""
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()

		if USE_AMP:
			with torch.amp.autocast("cuda", dtype=torch.bfloat16):
				outputs = model(images)
				loss = criterion(outputs, labels)
		else:
			outputs = model(images)
			loss = criterion(outputs, labels)

		loss.backward()
		clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
		optimizer.step()

		total_loss += loss.item() * images.size(0)
		correct += (outputs.argmax(1) == labels).sum().item()
		total += images.size(0)

	return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
	"""Evaluates one epoch. No AMP."""
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)

			total_loss += loss.item() * images.size(0)
			correct += (outputs.argmax(1) == labels).sum().item()
			total += images.size(0)

	return total_loss / total, correct / total


def main():
	total_start = time.time()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print(f"Device: {device}")
	print(f"Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {'bfloat16' if USE_AMP else 'off (fp32)'}")

	# dataset and class names
	train_data = FruitDataset(DATA_DIR, image_size=IMAGE_SIZE, augment=True)
	eval_data = FruitDataset(DATA_DIR, image_size=IMAGE_SIZE, augment=False)

	class_names = train_data.class_names
	num_classes = len(class_names)
	print(f"Classes ({num_classes}): {class_names}")
	print(f"Total images: {len(train_data)}")

	# 80/10/10 split with fixed random seed for reproducibility.
	n = len(train_data)
	perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
	n_train = int(0.8 * n)
	n_val = int(0.1 * n)

	train_idx = perm[:n_train]
	val_idx = perm[n_train:n_train + n_val]
	test_idx = perm[n_train + n_val:]

	train_subset = Subset(train_data, train_idx.tolist())
	val_subset = Subset(eval_data, val_idx.tolist())
	test_subset = Subset(eval_data, test_idx.tolist())

	print(f"Split: {len(train_subset)} train / {len(val_subset)} val / {len(test_subset)} test")

	# WeightedRandomSampler to address class imbalance in training set.
	train_labels = [train_data.class_to_idx[train_data.samples[i][1]] for i in train_idx.tolist()]
	class_counts = {}
	for label in train_labels:
		class_counts[label] = class_counts.get(label, 0) + 1
	sample_weights = [1.0 / class_counts[label] for label in train_labels]
	sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

	# DataLoaders
	train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
	val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	# model, criterion, optimizer
	model = build_model(num_classes)
	model.to(device)
	criterion = nn.CrossEntropyLoss()

	# resume training if checkpoint exists
	best_val_loss = float("inf")
	resume_ckpt = None

	try:
		ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
		answer = input("Resume training? [y/N]: ").strip().lower()
		if answer == "y":
			if ckpt.get("class_names") and ckpt["class_names"] != class_names:
				print(f"Class mismatch — checkpoint has {ckpt['class_names']} but dataset has {class_names}. Aborting.")
				return
			model.load_state_dict(ckpt["model_state"])
			best_val_loss = ckpt["best_val_loss"]
			resume_ckpt = ckpt
			print(f"Resuming from Phase {ckpt['phase']}, Epoch {ckpt['epoch'] + 1}")
		else:
			try:
				os.remove(RESUME_PATH)
			except FileNotFoundError:
				pass
			print("Starting fresh.")
	except FileNotFoundError:
		pass

	# phase 1: head only
	if resume_ckpt is None or resume_ckpt["phase"] == 1:
		for p in model.parameters():
			p.requires_grad = False
		for p in model.head.parameters():
			p.requires_grad = True

		optimizer = torch.optim.Adam(model.head.parameters(), lr=PHASE1_LR)

		start_epoch = 0
		if resume_ckpt and resume_ckpt["phase"] == 1:
			optimizer.load_state_dict(resume_ckpt["optimizer_state"])
			start_epoch = resume_ckpt["epoch"] + 1

		print(f"\n=== Phase 1: Head only ({PHASE1_EPOCHS} epochs, lr={PHASE1_LR}) ===")

		try:
			for epoch in range(start_epoch, PHASE1_EPOCHS):
				t0 = time.time()
				train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
				val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
				elapsed = time.time() - t0

				print(f"  Epoch {epoch+1:2d}/{PHASE1_EPOCHS:2d} | "
					  f"Train  loss={train_loss:.4f}  acc={train_acc:.1%} | "
					  f"Val  loss={val_loss:.4f}  acc={val_acc:.1%} | "
					  f"{_fmt_time(elapsed)}")

				if val_loss < best_val_loss:
					best_val_loss = val_loss
					save_model(model, class_names, MODEL_PATH)

				_save_resume(1, epoch, model, optimizer, None, best_val_loss, class_names)
		except KeyboardInterrupt:
			print("\nTraining interrupted. Resume checkpoint saved.")
			return

	# phase 2: full fine-tune
	for p in model.parameters():
		p.requires_grad = True

	optimizer = torch.optim.Adam(model.parameters(), lr=PHASE2_LR)
	scheduler = CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

	start_epoch = 0
	if resume_ckpt and resume_ckpt["phase"] == 2:
		optimizer.load_state_dict(resume_ckpt["optimizer_state"])
		if resume_ckpt.get("scheduler_state"):
			scheduler.load_state_dict(resume_ckpt["scheduler_state"])
		start_epoch = resume_ckpt["epoch"] + 1

	print(f"\n=== Phase 2: Full fine-tune ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR}) ===")

	try:
		for epoch in range(start_epoch, PHASE2_EPOCHS):
			t0 = time.time()
			train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
			val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
			current_lr = optimizer.param_groups[0]["lr"]
			scheduler.step()
			elapsed = time.time() - t0
			print(f"  Epoch {epoch+1:2d}/{PHASE2_EPOCHS:2d} | "
				  f"Train  loss={train_loss:.4f}  acc={train_acc:.1%} | "
				  f"Val  loss={val_loss:.4f}  acc={val_acc:.1%} | "
				  f"lr={current_lr:.2e} | "
				  f"{_fmt_time(elapsed)}")

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				save_model(model, class_names, MODEL_PATH)

			_save_resume(2, epoch, model, optimizer, scheduler, best_val_loss, class_names)
	except KeyboardInterrupt:
		print("\nTraining interrupted. Resume checkpoint saved.")
		return

	# cleanup resume checkpoint
	try:
		os.remove(RESUME_PATH)
	except FileNotFoundError:
		pass

	# final test evaluation with best model
	print("\n=== Final Test Evaluation ===")
	model, class_names = load_model(MODEL_PATH, device)
	model.eval()
	test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
	print(f"  Test  loss={test_loss:.4f}  acc={test_acc:.1%}")

	print(f"\nTotal training time: {_fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
	main()
