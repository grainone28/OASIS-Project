"""
data/cityscapes.py
PyTorch Dataset wrapper for Cityscapes (19-class semantic / panoptic split).
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ── Cityscapes 19-class label mapping ────────────────────────────────────────
# Maps raw trainId pixel values (0-18) to human-readable names.
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# Raw label IDs → trainId (255 = ignore)
LABEL_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,   9: 255, 10: 255, 11: 2,  12: 3,  13: 4,
    14: 255, 15: 255, 16: 255, 17: 5,  18: 255, 19: 6,  20: 7,
    21: 8,  22: 9,  23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255,
}


class CityscapesDataset(Dataset):
    """
    Loads Cityscapes images and ground-truth semantic masks.

    Expected directory layout:
        root/
          leftImg8bit/{split}/{city}/*.png
          gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str,
        split: str = "val",          # "train" | "val" | "test"
        joint_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.target_transform = target_transform

        self.images, self.masks = self._collect_paths()
        if len(self.images) == 0:
            raise FileNotFoundError(
                f"No images found for split='{split}' under {root}.\n"
                "Make sure leftImg8bit/ and gtFine/ exist."
            )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _collect_paths(self):
        img_dir = self.root / "leftImg8bit" / self.split
        lbl_dir = self.root / "gtFine" / self.split

        images, masks = [], []
        for city in sorted(img_dir.iterdir()):
            for img_path in sorted(city.glob("*_leftImg8bit.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                lbl_path = lbl_dir / city.name / f"{stem}_gtFine_labelIds.png"
                if lbl_path.exists():
                    images.append(img_path)
                    masks.append(lbl_path)

        return images, masks

    # Convert raw Cityscapes labelIds to the 19-class trainId space.
    # Unused / void classes are mapped to IGNORE_INDEX = 255.
    @staticmethod
    def _encode_target(mask_pil: Image.Image) -> torch.Tensor:
        """Convert raw labelId mask → trainId tensor (uint8)."""
        arr = np.array(mask_pil, dtype=np.int32)
        out = np.full_like(arr, fill_value=255, dtype=np.uint8)
        for raw_id, train_id in LABEL_TO_TRAINID.items():
            out[arr == raw_id] = train_id
        return torch.from_numpy(out).long()

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)
    
    # NOTE:
    # The dataset now supports joint image-mask transforms.
    # In the previous version, image and target transforms were independent,
    # which was unsafe for random geometric augmentations.
    # We now apply joint transforms first, then optional image-only transforms,
    # and finally encode the mask into Cityscapes trainIds.
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        # geometric transforms applied to both
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # extra transforms only on image or mask if needed
        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask_tensor = self._encode_target(mask)
        return image, mask_tensor

    def get_image_path(self, idx: int) -> str:
        return str(self.images[idx])
