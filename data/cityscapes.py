"""
data/cityscapes.py
PyTorch Dataset wrapper for Cityscapes (19-class semantic / panoptic split).

──────────────────────────────────────────────────────────────────────────────
MODIFICHE (Step 4 — confronto COCO vs Cityscapes-finetuned)
──────────────────────────────────────────────────────────────────────────────
Aggiunte rispetto alla versione originale (le righe sopra `class CityscapesDataset`):

  • COCO_PANOPTIC_CLASSES     — lista dei 133 nomi classi di COCO-Panoptic
                                (80 things + 53 stuff), nell'ordine standard
                                detectron2 / panopticapi.

  • COCO_TO_CITYSCAPES        — dizionario {coco_id: cityscapes_trainId}
                                con 21 mapping. Le classi non presenti
                                vengono trattate come ignore (255).

  • build_coco_to_cityscapes_lut(num_coco_classes=133, ignore_index=255)
                              — costruisce un tensore LUT (133,) per
                                applicare il remap in modo vettorizzato.

  • remap_coco_to_cityscapes(pred, lut=None, ignore_index=255)
                              — applica la LUT a una mappa di predizioni
                                (H, W) o (B, H, W); gestisce id fuori range.

  • eomt_to_semantic_logits(pred_logits, pred_masks)
                              — converte l'output mask-classification di EoMT
                                (B, Q, C+1) + (B, Q, H, W) → logits semantici
                                (B, C, H, W) usando la formula marginale
                                standard softmax(cls) · sigmoid(mask).

Il blocco `LABEL_TO_TRAINID` e la classe `CityscapesDataset` non sono stati
modificati: la pipeline di caricamento delle ground truth Cityscapes resta
identica.
──────────────────────────────────────────────────────────────────────────────
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


# ── COCO-Panoptic (133 classi) → Cityscapes (19 trainId) ─────────────────────
# Necessario per lo Step 4: il modello eomt_coco predice 133 classi
# (80 things + 53 stuff), Cityscapes ne ha 19. Senza remap, calcolare la
# mIoU crasha per incompatibilità di dimensioni o gonfia falsi positivi.
#
# Assunzione sull'ordinamento: COCO-Panoptic contiguous IDs nella convenzione
# di detectron2 / panopticapi — things 0..79, stuff 80..132. Se il tuo
# checkpoint usa un ordinamento diverso, stampa `model.class_embed.out_features`
# e l'eventuale id2label e aggiorna sotto.
COCO_PANOPTIC_CLASSES = [
    # Things (0-79)
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
    # Stuff (80-132)
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

# Mapping COCO contiguous id → Cityscapes trainId (0-18).
# Le classi non presenti qui vengono trattate come ignore (255).
# Note sui compromessi inevitabili:
#   - "rider" (cs 12) non esiste in COCO: una persona su bici/moto viene
#     predetta come "person" e finisce in cs 11. Limite di dominio.
#   - "pole" (cs 5) non ha equivalente affidabile → ignore.
#   - "traffic sign" è approssimato con "stop sign" (l'unico segnale stradale
#     in COCO); tutti gli altri segnali stradali non saranno predetti.
#   - "terrain" è approssimato con "grass-merged".
COCO_TO_CITYSCAPES = {
    # Things
    0:  11,   # person       → person
    1:  18,   # bicycle      → bicycle
    2:  13,   # car          → car
    3:  17,   # motorcycle   → motorcycle
    5:  15,   # bus          → bus
    6:  16,   # train        → train
    7:  14,   # truck        → truck
    9:   6,   # traffic light→ traffic light
    11:  7,   # stop sign    → traffic sign (approssimazione)
    # Stuff
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
    """
    Costruisce un tensore LUT di forma (num_coco_classes,) tale che
    `lut[coco_id]` restituisca il corrispondente trainId Cityscapes
    (o `ignore_index` se la classe COCO non ha mapping).

    Uso tipico — una singola operazione di indicizzazione vettorizzata
    sulla mappa di predizioni:

        lut = build_coco_to_cityscapes_lut().to(device)
        cs_pred = lut[coco_pred]    # broadcasting su (B, H, W)
    """
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
    """
    Rimappa una mappa di predizioni nello spazio classi COCO (0..132)
    allo spazio classi Cityscapes (0..18, oppure 255 per ignore).

    Args:
        pred: tensor di interi con id classe COCO, forma (H, W) o (B, H, W).
        lut:  LUT pre-calcolata. Se None, viene costruita on-the-fly
              (preferisci pre-calcolarla fuori dal loop).

    Returns:
        Tensor della stessa forma di `pred`, con trainId Cityscapes (0-18)
        oppure `ignore_index` (255) per le classi COCO senza corrispondenza.
    """
    if lut is None:
        lut = build_coco_to_cityscapes_lut(ignore_index=ignore_index)
    lut = lut.to(pred.device)

    # Clamp difensivo contro id fuori range; poi ri-iniettiamo ignore.
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
    """
    Converte l'output mask-classification di EoMT (queries × maschere)
    in una mappa di logits semantici per-pixel, pronta per argmax.

    Usa la formula marginale standard (Mask2Former / Panoptic-FPN):

        sem[b, c, h, w] = Σ_q softmax(cls_logits[b, q, :C])[c] · sigmoid(mask[b, q, h, w])

    L'ultima classe di `pred_logits` ("no object") viene scartata prima
    della softmax.

    Args:
        pred_logits: (B, Q, C+1)
        pred_masks:  (B, Q, H, W)

    Returns:
        (B, C, H, W) — non sono vere probabilità (somma su c ≠ 1) ma sono
        monotonicamente coerenti per argmax, che è ciò che serve.
    """
    mask_cls  = pred_logits.softmax(dim=-1)[..., :-1]   # (B, Q, C)
    mask_pred = pred_masks.sigmoid()                     # (B, Q, H, W)
    return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask_tensor = self._encode_target(mask)
        return image, mask_tensor

    def get_image_path(self, idx: int) -> str:
        return str(self.images[idx])
