"""
dataset.py — Loads fruit images from disk and prepares them for PyTorch training.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Calculated by averaging every pixel across 1.2M ImageNet training images.
# These values are a universal standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB means
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB standard deviations

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}  # Supported image file extensions


def get_val_transform(image_size=224):
	"""Returns the validation/inference transform: resize, to tensor, and normalize."""
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])


class FruitDataset(Dataset):
    """
    Loads images from the nested data/Train structure:

        data/Train/apple/fresh/   -> class "apple_fresh"
        data/Train/apple/rotten/  -> class "apple_rotten"
        data/Train/banana/fresh/  -> class "banana_fresh"
        ...
        data/Train/unknown/       -> class "unknown"  (flat, no subdirs)

    Returns (image_tensor, class_index) pairs.
    """

    def __init__(self, data_dir: str | Path, image_size: int = 224, augment: bool = False):
        """
        Args:
            data_dir:   Path to the data root folder ("data/Train")
            image_size: Resize all images to this square size.
                        DINOv2 ViT-B/14 is natively pretrained at 224x224.
            augment:    If True, apply random augmentations (this should only be used for training).
        """
        self.data_dir = Path(data_dir) # Convert to Path object for easier path manipulation

        # ToTensor converts the PIL image to a float tensor and scales pixel values to [0,1].
        # Normalize then shifts them to match the distribution the model was pretrained on. Without this, predictions are poor.
        to_tensor_and_norm = [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        if augment:
            # Model sees a slightly different version of every image making it harder
            # to memorize and forcing it to learn features that hold up across real-world variation.
            # Randomly crops and scales, flips horizontally, rotates up to 15°, and shifts brightness, contrast, saturation, and hue slightly.
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            ] + to_tensor_and_norm)
        else:
            self.transform = get_val_transform(image_size)

        self.samples: List[Tuple[Path, str]] = [] # list of (image_path, class_label_string) pairs
        class_name_set: set = set() # list of unique class label strings

        for fruit_dir in sorted(self.data_dir.iterdir()): # Iterate over subdirectories
            if not fruit_dir.is_dir(): # Skip files (we want dirs only)
                continue

            # Special case for "unknown/" dir
            if fruit_dir.name.lower() == "unknown":
                label = "unknown"
                class_name_set.add(label)
                for img_path in fruit_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                        self.samples.append((img_path, label))
            else:
                # Standard case fruit/fresh/ and fruit/rotten/ subdirectories.
                for freshness_dir in sorted(fruit_dir.iterdir()):
                    if (not freshness_dir.is_dir()
                            or freshness_dir.name.lower() not in {"fresh", "rotten"}):
                        continue

                    # Label format: "fruit_freshness" ("apple_fresh")
                    label = f"{fruit_dir.name}_{freshness_dir.name.lower()}"
                    class_name_set.add(label)

                    # Find all files recursively inside the folder
                    for img_path in freshness_dir.rglob("*"):
                        if img_path.suffix.lower() in IMG_EXTS:
                            self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No images found under: {self.data_dir}")

        # Sort class names alphabetically and assign each a unique integer index.
        self.class_names = sorted(class_name_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)} #

    def __len__(self) -> int:
        # Returns total dataset size (image_path, class_label) pairs.
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns one (image_tensor, class_index) pair.
        DataLoader calls this repeatedly with different indices.
        """
        img_path, label = self.samples[idx]

        # Convert to RGB ensures 3 channels, no transparency.
        img = Image.open(img_path).convert("RGB")

        # Apply transforms and convert to tensor (resize, augmentations, to tensor, normalize)
        tensor = self.transform(img)

        return tensor, self.class_to_idx[label]