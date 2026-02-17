from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}

class FruitDataset(Dataset):
    # Constructor
    def __init__(self, data_dir: str | Path, image_size: int = 224, augment: bool = False):
        self.data_dir = Path(data_dir)

        # Create transforms based on augmentation setting
        if augment:
            # Training transforms: apply random augmentations
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            # Validation/inference transforms: no augmentation
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        # Find all images and their labels
        self.samples: List[Tuple[Path, str]] = []
        self.class_names: List[str] = []
        
        # Traverse the directory structure
        for fruit_dir in sorted(self.data_dir.iterdir()):
            if not fruit_dir.is_dir():
                continue
            for freshness_dir in sorted(fruit_dir.iterdir()):
                # Only consider directories named "fresh" or "rotten" (case-insensitive)
                if not freshness_dir.is_dir() or freshness_dir.name.lower() not in {"fresh", "rotten"}:
                    continue
                
                # Create label in the format "fruit_freshness"
                label = f"{fruit_dir.name}_{freshness_dir.name.lower()}"

                # Add label to class names if it's not already there
                if label not in self.class_names:
                    self.class_names.append(label)
                
                # Traverse the freshness directory to find image files
                for img_path in freshness_dir.rglob("*"):
                    if img_path.suffix.lower() in IMG_EXTS:
                        self.samples.append((img_path, label))
        
        if not self.samples:
            raise ValueError(f"No images found under: {self.data_dir}")
        
        # Create a mapping from class names to indices
        self.class_to_idx = {c: i for i, c in enumerate(sorted(self.class_names))}

    def __len__(self) -> int:
        return len(self.samples)

    # Get an item by index
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), self.class_to_idx[label]