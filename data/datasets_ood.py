"""
data/datasets_ood.py
Dataset wrappers for Out-of-Distribution benchmarks:
  - Fishyscapes Lost & Found
  - SegmentMeIfYouCan (SMIYC) — RoadAnomaly21 & RoadObstacle21
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class FishyscapesLostAndFound(Dataset):
    """
    Fishyscapes Lost & Found validation set.

    Directory layout expected:
        root/
          images/*.png
          labels/*.png      # 0 = in-distribution, 1 = anomaly, 255 = ignore
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.images = sorted((self.root / "images").glob("*.png"))
        self.labels = sorted((self.root / "labels").glob("*.png"))
        assert len(self.images) == len(self.labels), "Image/label count mismatch."

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = np.array(Image.open(self.labels[idx]))  # {0, 1, 255}

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.from_numpy(label).long()
        return image, label_tensor


class SMIYCDataset(Dataset):
    """
    SegmentMeIfYouCan benchmark.
    Supports 'RoadAnomaly21' and 'RoadObstacle21' subsets.

    Directory layout expected:
        root/
          images/validation/*.jpg  (or .png)
          labels_masks/validation/*.png   # 0 = road, 1 = anomaly, 255 = ignore
    """

    def __init__(
        self,
        root: str,
        subset: str = "RoadAnomaly21",   # or "RoadObstacle21"
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root) / subset
        self.transform = transform

        img_dir = self.root / "images" / "validation"
        lbl_dir = self.root / "labels_masks" / "validation"

        self.images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        self.labels = sorted(lbl_dir.glob("*.png"))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found under {img_dir}")
        if len(self.images) != len(self.labels):
            raise AssertionError(
                f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels."
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = np.array(Image.open(self.labels[idx]))

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.from_numpy(label).long()
        return image, label_tensor
