"""
evaluate_miou.py — Step 4: Evaluation Baseline
================================================
Evaluate an EoMT (or ERFNet) checkpoint on the Cityscapes validation set
and compute mean Intersection-over-Union (mIoU).

Usage:
  python evaluate_miou.py --config configs/default_eomt.yaml \
                          --checkpoint checkpoints/eomt_cityscapes.pth \
                          --model eomt

  python evaluate_miou.py --config configs/default_eomt.yaml \
                          --checkpoint checkpoints/erfnet_cityscapes.pth \
                          --model erfnet
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cityscapes import (
    CityscapesDataset,
    build_coco_to_cityscapes_lut,
    remap_coco_to_cityscapes,
)
from data.transforms import get_val_transform, get_mask_transform
from utils.metrics import MeanIoUMeter
from utils.logger import setup_logging


def build_model(
    model_name: str,
    cfg: dict,
    checkpoint: str,
    device: torch.device,
    num_classes_override: Optional[int] = None,
):
    # Il numero di classi del *modello* può differire da quello del dataset
    # (es. checkpoint COCO con 133 classi su Cityscapes con 19 trainId).
    num_classes = num_classes_override or cfg["dataset"]["num_classes"]

    if model_name == "eomt":
        from models.eomt import EoMT
        model = EoMT(
            backbone_name  = cfg["model"].get("backbone", "vit_base_patch14_dinov2"),
            num_queries    = cfg["model"].get("num_queries", 100),
            num_classes    = num_classes,
            freeze_layers  = cfg["model"].get("freeze_backbone_layers", 9),
            image_size     = tuple(cfg["dataset"]["image_size"]),
        )
    elif model_name == "erfnet":
        from models.erfnet import ERFNet
        model = ERFNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
        if missing:
            print(f"  [warn] missing keys   : {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, dataloader, device, model_name, num_classes, is_coco=False):
    meter = MeanIoUMeter(num_classes=num_classes)

    # LUT COCO → Cityscapes calcolata una volta sola se serve
    coco_lut = build_coco_to_cityscapes_lut().to(device) if is_coco else None

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images  = images.to(device)
        targets = targets.to(device)

        if model_name == "eomt":
            outputs = model(images)
            mask_logits  = outputs["pred_masks"]    # [B, Q, H, W]
            class_logits = outputs["pred_logits"]   # [B, Q, C+1]

            # Semantic inference (Mask2Former-style): per ogni pixel
            # combina mask-prob e class-prob su tutte le query, poi argmax.
            #   sem_logits[b,c,h,w] = Σ_q σ(mask[b,q,h,w]) · softmax(cls[b,q])[c]
            # Confrontato con il "best query → its class", questo sfrutta
            # tutte le query e dà mappe più pulite.
            mask_probs  = torch.sigmoid(mask_logits)                       # [B, Q, H, W]
            class_probs = torch.softmax(class_logits, dim=-1)[..., :-1]    # [B, Q, C]
            sem_logits  = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
            preds = sem_logits.argmax(dim=1)        # [B, H, W] — id nello spazio del modello

            if is_coco:
                # Rimappa dallo spazio COCO-Panoptic (133) allo spazio Cityscapes (19).
                # Le classi COCO senza corrispondente diventano 255 e vengono
                # ignorate dal MeanIoUMeter.
                preds = remap_coco_to_cityscapes(preds, lut=coco_lut)

        else:  # ERFNet
            logits = model(images)                     # [B, C, H, W]
            preds  = logits.argmax(dim=1)              # [B, H, W]

        # Resize predictions to match target resolution if needed
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(
                preds.float().unsqueeze(1), size=targets.shape[-2:], mode="nearest"
            ).squeeze(1).long()

        meter.update(preds, targets)

    return meter.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation mIoU")
    parser.add_argument("--config",     default="configs/default_eomt.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model",      choices=["eomt", "erfnet"], default="eomt")
    parser.add_argument("--is-coco",    action="store_true",
                        help="Il checkpoint è in spazio COCO-Panoptic; "
                             "applica remap COCO→Cityscapes prima della mIoU.")
    parser.add_argument("--coco-num-classes", type=int, default=133,
                        help="Numero di classi del checkpoint COCO "
                             "(133 = COCO-Panoptic, 80 = COCO-Instance).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    logger = setup_logging()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = CityscapesDataset(
        root=cfg["dataset"]["root"],
        split="val",
        transform=get_val_transform(tuple(cfg["dataset"]["image_size"])),
        target_transform=get_mask_transform(tuple(cfg["dataset"]["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # Se è un checkpoint COCO, il modello DEVE essere costruito con le classi
    # COCO, non con 19 — altrimenti la testa class_embed non corrisponde
    # e load_state_dict(strict=False) la lascia con pesi random.
    model_num_classes = args.coco_num_classes if args.is_coco else None
    model = build_model(args.model, cfg, args.checkpoint, device,
                        num_classes_override=model_num_classes)

    logger.info(f"Running evaluation: model={args.model}, COCO-remap={args.is_coco}")
    # La mIoU si calcola SEMPRE nello spazio Cityscapes (19 classi).
    results = evaluate(model, loader, device, args.model,
                       cfg["dataset"]["num_classes"], is_coco=args.is_coco)

    logger.info(f"mIoU     : {results['mIoU']:.4f}")
    logger.info(f"Pixel Acc: {results['pixel_acc']:.4f}")
    from data.cityscapes import CITYSCAPES_CLASSES
    for name, iou in zip(CITYSCAPES_CLASSES, results["per_class_iou"]):
        logger.info(f"  {name:>14s}: {iou:.4f}")


if __name__ == "__main__":
    main()
