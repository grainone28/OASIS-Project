import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

#  Cityscapes 19-class label mapping 

# Maps raw trainId pixel values (0-18)
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


# COCO-Panoptic (133) --> Cityscapes (19) 

COCO_PANOPTIC_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
    "banner", "blanket", "bridge", "cardboard", "counter", "curtain",
    "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light",
    "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad",
    "river", "road", "roof", "sand", "sea", "shelf", "snow", "stairs", "tent",
    "towel", "wall-brick", "wall-stone", "wall-tile", "wall-wood",
    "water-other", "window-blind", "window-other", "tree-merged",
    "fence-merged", "ceiling-merged", "sky-other-merged", "cabinet-merged",
    "table-merged", "floor-other-merged", "pavement-merged",
    "mountain-merged", "grass-merged", "dirt-merged", "paper-merged",
    "food-other-merged", "building-other-merged", "rock-merged",
    "wall-other-merged", "rug-merged",
]


COCO_TO_CITYSCAPES = {
    0:  11,   # person       → person
    1:  18,   # bicycle      → bicycle
    2:  13,   # car          → car
    3:  17,   # motorcycle   → motorcycle
    5:  15,   # bus          → bus
    6:  16,   # train        → train
    7:  14,   # truck        → truck
    9:   6,   # traffic light→ traffic light
    11:  7,   # stop sign    → traffic sign

    100: 0,   # road                  → road
    109: 3,   # wall-brick            → wall
    110: 3,   # wall-stone            → wall
    111: 3,   # wall-tile             → wall
    112: 3,   # wall-wood             → wall
    116: 8,   # tree-merged           → vegetation
    117: 4,   # fence-merged          → fence
    119: 10,  # sky-other-merged      → sky
    123: 1,   # pavement-merged       → sidewalk
    125: 9,   # grass-merged          → terrain
    129: 2,   # building-other-merged → building
    131: 3,   # wall-other-merged     → wall
}


def build_coco_to_cityscapes_lut(
    num_coco_classes: int = 133,
    ignore_index: int = 255,
) -> torch.Tensor:
    
    lut = torch.full((num_coco_classes,), ignore_index, dtype=torch.long)
    for coco_id, cs_id in COCO_TO_CITYSCAPES.items():
        if 0 <= coco_id < num_coco_classes:
            lut[coco_id] = cs_id
    return lut


def remap_coco_to_cityscapes(
    pred: torch.Tensor,
    lut: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    
    if lut is None:
        lut = build_coco_to_cityscapes_lut(ignore_index=ignore_index)
    lut = lut.to(pred.device)

    pred_safe = pred.clamp(min=0, max=lut.numel() - 1)
    remapped = lut[pred_safe]
    out_of_range = (pred < 0) | (pred >= lut.numel())
    if out_of_range.any():
        remapped[out_of_range] = ignore_index
    return remapped


def eomt_to_semantic_logits(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
) -> torch.Tensor:
    
    mask_cls  = pred_logits.softmax(dim=-1)[..., :-1]   
    mask_pred = pred_masks.sigmoid()                     
    return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)


class CityscapesDataset(Dataset):

    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str,
        split: str = "val",          # "train"  "val"  "test"
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.images, self.masks = self._collect_paths()
        if len(self.images) == 0:
            raise FileNotFoundError(
                f"No images found for split='{split}' under {root}.\n"
                "Make sure leftImg8bit/ and gtFine/ exist."
            )

    #  helpers 

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

    @staticmethod
    def _encode_target(mask_pil: Image.Image) -> torch.Tensor:
        arr = np.array(mask_pil, dtype=np.int32)
        out = np.full_like(arr, fill_value=255, dtype=np.uint8)
        for raw_id, train_id in LABEL_TO_TRAINID.items():
            out[arr == raw_id] = train_id
        return torch.from_numpy(out).long()

    # Dataset interface 

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        mask  = Image.open(self.masks[idx])

        if self.transform:
            image = self.transform(image)

        mask_tensor = self._encode_target(mask)   

        if self.target_transform:
            mask_tensor = self.target_transform(mask_tensor)

        return image, mask_tensor

    def get_image_path(self, idx: int) -> str:
        return str(self.images[idx])
