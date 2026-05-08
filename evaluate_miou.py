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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cityscapes import CityscapesDataset
from data.transforms import get_val_transform, get_mask_transform
from utils.metrics import MeanIoUMeter
from utils.logger import setup_logging


# ── COCO → Cityscapes class index remapping (Step 4 key challenge) ───────────
# EoMT-COCO outputs 80 COCO classes; we map the 19 Cityscapes trainIds
# to the corresponding COCO category index.
CITYSCAPES_TO_COCO = {
    0: 43,   # road → pavement
    1: 49,   # sidewalk → pavement (approximate)
    11: 0,   # person → person
    13: 2,   # car → car
    14: 7,   # truck → truck
    15: 5,   # bus → bus
    18: 1,   # bicycle → bicycle
    # Classes with no COCO equivalent will have mIoU ≈ 0
}


def build_model(model_name: str, cfg: dict, checkpoint: str, device: torch.device):
    if model_name == "eomt":
        from models.eomt import build_eomt
        model = build_eomt(cfg)
    elif model_name == "erfnet":
        from models.erfnet import ERFNet
        model = ERFNet(num_classes=cfg["dataset"]["num_classes"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, dataloader, device, model_name, num_classes, is_coco=False):
    meter = MeanIoUMeter(num_classes=num_classes)

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images  = images.to(device)
        targets = targets.to(device)

        if model_name == "eomt":
            outputs = model(images)
            logits = outputs["pred_masks"]             # [B, Q, H, W]
            class_logits = outputs["pred_logits"]      # [B, Q, C+1]

            # Convert mask-based output → per-pixel class prediction
            # For each pixel, find the query with highest mask score,
            # then assign that query's predicted class.
            mask_probs  = torch.sigmoid(logits)        # [B, Q, H, W]
            class_probs = torch.softmax(class_logits[..., :-1], dim=-1)  # [B, Q, C]
            class_preds = class_probs.argmax(dim=-1)   # [B, Q]

            B, Q, H, W = mask_probs.shape
            best_query  = mask_probs.argmax(dim=1)     # [B, H, W]
            preds = class_preds[
                torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W),
                best_query
            ]                                          # [B, H, W]

            if is_coco:
                # Remap COCO prediction indices to Cityscapes trainIds
                remapped = torch.full_like(preds, fill_value=255)
                for cs_id, coco_id in CITYSCAPES_TO_COCO.items():
                    remapped[preds == coco_id] = cs_id
                preds = remapped

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
                        help="Use COCO→Cityscapes class remapping")
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

    model = build_model(args.model, cfg, args.checkpoint, device)

    logger.info(f"Running evaluation: model={args.model}, COCO-remap={args.is_coco}")
    results = evaluate(model, loader, device, args.model,
                       cfg["dataset"]["num_classes"], is_coco=args.is_coco)

    logger.info(f"mIoU     : {results['mIoU']:.4f}")
    logger.info(f"Pixel Acc: {results['pixel_acc']:.4f}")
    for i, iou in enumerate(results["per_class_iou"]):
        logger.info(f"  Class {i:02d}: {iou:.4f}")


if __name__ == "__main__":
    main()
