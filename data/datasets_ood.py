from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _build_pairs(image_paths: List[Path], label_paths: List[Path]):
    # Pair images and labels by sorted order.  
    if len(image_paths) == 0:
        raise FileNotFoundError("No images found.")
    if len(image_paths) != len(label_paths):
        raise AssertionError(
            f"Mismatch: {len(image_paths)} images vs {len(label_paths)} labels."
        )
    return list(zip(sorted(image_paths), sorted(label_paths)))


def _to_label_tensor(label_pil: Image.Image) -> torch.Tensor:
    # Convert anomaly mask to torch.long tensor. 
    label = np.array(label_pil)
    label = np.where(label == 2, 1, label)
    return torch.from_numpy(label).long()


class FishyscapesLostAndFound(Dataset):
    # Fishyscapes Lost & Found validation set.
    def __init__(
        self,
        root: str,
        joint_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.joint_transform = joint_transform
        self.image_transform = image_transform

        image_paths = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            image_paths += list(self.root.rglob(f"images/**/*.{ext}"))
        image_paths = sorted(image_paths)

        label_paths = sorted(list(self.root.rglob("labels/**/*.png")))
        if not label_paths:
            label_paths = sorted(list(self.root.rglob("labels_masks/**/*.png")))

        print(f"[DEBUG] Found {len(image_paths)} images and {len(label_paths)} labels.")
        self.samples = _build_pairs(image_paths, label_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        if self.image_transform:
            image = self.image_transform(image)

        label_tensor = _to_label_tensor(label)

        H, W = image.shape[-2], image.shape[-1]
        if label_tensor.shape != torch.Size([H, W]):
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="nearest"
            ).squeeze().long()

        return image, label_tensor

    def get_sample_paths(self, idx: int) -> Tuple[str, str]:
        image_path, label_path = self.samples[idx]
        return str(image_path), str(label_path)


class SMIYCDataset(Dataset):
    # SegmentMeIfYouCan benchmark.
    def __init__(
        self,
        root: str,
        subset: str = "RoadAnomaly21",
        joint_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
    ):
        self.root = Path(root) / subset
        self.subset = subset
        self.joint_transform = joint_transform
        self.image_transform = image_transform

        img_dir = self.root / "images" / "validation"
        lbl_dir = self.root / "labels_masks" / "validation"

        image_paths = (
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.png")) +
            list(img_dir.glob("*.webp"))
        )
        label_paths = list(lbl_dir.glob("*.png"))

        if len(image_paths) == 0:
            raise FileNotFoundError(
                f"No images found under {img_dir}. "
                f"Please check that the folder exists and contains .jpg, .png, or .webp files. "
                f"Folder content: {list(img_dir.iterdir())[:5] if img_dir.exists() else 'Folder does not exist'}"
            )

        self.samples = _build_pairs(image_paths, label_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        if self.image_transform:
            image = self.image_transform(image)

        label_tensor = _to_label_tensor(label)

        H, W = image.shape[-2], image.shape[-1]
        if label_tensor.shape != torch.Size([H, W]):
            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W),
                mode="nearest"
            ).squeeze().long()

        return image, label_tensor

    def get_sample_paths(self, idx: int) -> Tuple[str, str]:
        image_path, label_path = self.samples[idx]
        return str(image_path), str(label_path)