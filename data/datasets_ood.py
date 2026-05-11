"""
data/datasets_ood.py
Dataset wrappers for Out-of-Distribution anomaly segmentation benchmarks:
  - Fishyscapes Lost & Found
  - SegmentMeIfYouCan (SMIYC): RoadAnomaly21 / RoadObstacle21

Output convention:
  image: torch.Tensor [3, H, W]
  label: torch.LongTensor [H, W]
Labels are expected to follow:
  0   -> in-distribution / normal
  1   -> anomaly
  255 -> ignore
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# NOTE:
# These helpers mirror the structure of CityscapesDataset:
# - explicit pairing of image/label paths
# - optional joint transform for geometric ops
# - image-only transform for normalization / photometric ops


def _build_pairs(image_paths: List[Path], label_paths: List[Path]):
    """
    Pair images and labels by sorted order.

    This assumes that the dataset folder structure already guarantees
    one-to-one correspondence between images/labels and that lexicographic
    sorting is consistent across both lists.
    """
    if len(image_paths) == 0:
        raise FileNotFoundError("No images found.")
    if len(image_paths) != len(label_paths):
        raise AssertionError(
            f"Mismatch: {len(image_paths)} images vs {len(label_paths)} labels."
        )
    return list(zip(sorted(image_paths), sorted(label_paths)))


def _to_label_tensor(label_pil: Image.Image) -> torch.Tensor:
    """
    Convert anomaly mask to torch.long tensor.
    Expected values are usually {0, 1, 255}.
    """
    label = np.array(label_pil)
    return torch.from_numpy(label).long()


class FishyscapesLostAndFound(Dataset):
    """
    Fishyscapes Lost & Found validation set.

    Expected layout:
        root/
          images/*.png
          labels/*.png

    Labels:
        0   -> in-distribution / normal
        1   -> anomaly
        255 -> ignore
    """

    def __init__(
        self,
        root: str,
        joint_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.joint_transform = joint_transform
        self.image_transform = image_transform

        img_dir = self.root / "images"
        lbl_dir = self.root / "labels"

        image_paths = list(img_dir.glob("*.png"))
        label_paths = list(lbl_dir.glob("*.png"))
        self.samples = _build_pairs(image_paths, label_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        # Geometric transforms applied to both
        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        # Photometric / normalization only on the image
        if self.image_transform:
            image = self.image_transform(image)

        label_tensor = _to_label_tensor(label)
        return image, label_tensor

    def get_sample_paths(self, idx: int) -> Tuple[str, str]:
        image_path, label_path = self.samples[idx]
        return str(image_path), str(label_path)


class SMIYCDataset(Dataset):
    """
    SegmentMeIfYouCan benchmark.

    Supported subsets:
      - RoadAnomaly21
      - RoadObstacle21

    Expected layout:
        root/
          RoadAnomaly21/
            images/validation/*.jpg or *.png
            labels_masks/validation/*.png
          RoadObstacle21/
            images/validation/*.jpg or *.png
            labels_masks/validation/*.png

    Labels:
        0   -> road / normal
        1   -> anomaly
        255 -> ignore
    """

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

        image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        label_paths = list(lbl_dir.glob("*.png"))

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found under {img_dir}")

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
        return image, label_tensor

    def get_sample_paths(self, idx: int) -> Tuple[str, str]:
        image_path, label_path = self.samples[idx]
        return str(image_path), str(label_path)