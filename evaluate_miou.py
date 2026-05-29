import argparse
import yaml
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cityscapes import (
    CityscapesDataset,
    build_coco_to_cityscapes_lut,
    remap_coco_to_cityscapes,
    CITYSCAPES_CLASSES,
)
from data.transforms import get_val_transform, get_val_transform_erfnet, get_mask_transform
from utils.metrics import MeanIoUMeter
from utils.logger import setup_logging


def _detect_eomt_orig_config(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    state = {k.replace("network.", "", 1): v for k, v in state.items()
             if not k.startswith("criterion")}
    n_patches = state["encoder.backbone.pos_embed"].shape[1]
    grid_size = int(n_patches ** 0.5)
    img_size = (grid_size * 16, grid_size * 16)
    num_q = state["q.weight"].shape[0]
    num_classes = state["class_head.weight"].shape[0] - 1
    return img_size, num_q, num_classes


def build_model(model_name, cfg, checkpoint, device, num_classes_override=None):
    num_classes = num_classes_override or cfg["dataset"]["num_classes"]

    if model_name == "eomt-orig":
        from training.models.eomt import EoMT as EoMTOrig
        from training.models.vit import ViT
        img_size_det, num_q_det, num_classes_det = _detect_eomt_orig_config(checkpoint)
        print(f"[EoMT-orig] Detected: img_size={img_size_det}, "
              f"num_q={num_q_det}, num_classes={num_classes_det}")
        vit = ViT(img_size=img_size_det, patch_size=16,
                  backbone_name="vit_base_patch14_reg4_dinov2")
        model = EoMTOrig(encoder=vit, num_classes=num_classes_det,
                         num_q=num_q_det, num_blocks=3, masked_attn_enabled=True)
    elif model_name == "erfnet":
        from training.models.erfnet import ERFNet
        model = ERFNet(num_classes=20 if num_classes_override is None else num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if model_name == "eomt-orig":
            state = {k.replace("network.", "", 1): v for k, v in state.items()
                     if not k.startswith("criterion")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")

    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, dataloader, device, model_name, num_classes, is_coco=False):
    meter = MeanIoUMeter(num_classes=num_classes)
    coco_lut = build_coco_to_cityscapes_lut().to(device) if is_coco else None

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images, targets = images.to(device), targets.to(device)

        if model_name == "eomt-orig":
            mask_logits_list, class_logits_list = model(images)
            mask_logits = F.interpolate(mask_logits_list[-1], size=images.shape[-2:],
                                        mode="bilinear", align_corners=False)
            class_logits = class_logits_list[-1]
            mask_probs = torch.sigmoid(mask_logits)
            class_probs = torch.softmax(class_logits, dim=-1)[..., :-1]
            sem_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
            preds = sem_logits.argmax(dim=1)
            if is_coco:
                preds = remap_coco_to_cityscapes(preds, lut=coco_lut)
        else:  
            logits = model(images)
            if logits.shape[1] > num_classes:
                logits = logits[:, :num_classes]
            preds = logits.argmax(dim=1)

        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds.float().unsqueeze(1), size=targets.shape[-2:],
                                  mode="nearest").squeeze(1).long()
        meter.update(preds, targets)

    return meter.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_eval.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model", choices=["eomt-orig", "erfnet"], default="eomt-orig")
    parser.add_argument("--is-coco", action="store_true")
    parser.add_argument("--coco-num-classes", type=int, default=133)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logger = setup_logging()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if "root" not in cfg["dataset"]:
        cfg["dataset"]["root"] = cfg["dataset"].get("cityscapes_root", "./data/cityscapes")
    if "num_classes" not in cfg["dataset"]:
        cfg["dataset"]["num_classes"] = 19
    if "image_size" not in cfg["dataset"]:
        cfg["dataset"]["image_size"] = [512, 1024]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "eomt-orig" and args.checkpoint:
        img_size_det, _, _ = _detect_eomt_orig_config(args.checkpoint)
        cfg["dataset"]["image_size"] = list(img_size_det)
        logger.info(f"[EoMT-orig] image_size from checkpoint: {img_size_det}")
    logger.info(f"Device: {device}")

    img_transform = get_val_transform_erfnet(tuple(cfg["dataset"]["image_size"]))

    dataset = CityscapesDataset(
        root=cfg["dataset"]["root"], split="val",
        transform=img_transform,
        target_transform=get_mask_transform(tuple(cfg["dataset"]["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    model_num_classes = args.coco_num_classes if args.is_coco else None
    model = build_model(args.model, cfg, args.checkpoint, device,
                        num_classes_override=model_num_classes)

    logger.info(f"Running evaluation: model={args.model}, COCO-remap={args.is_coco}")
    results = evaluate(model, loader, device, args.model,
                       cfg["dataset"]["num_classes"], is_coco=args.is_coco)

    logger.info(f"mIoU     : {results['mIoU']:.4f}")
    logger.info(f"Pixel Acc: {results['pixel_acc']:.4f}")
    for name, iou in zip(CITYSCAPES_CLASSES, results["per_class_iou"]):
        logger.info(f"  {name:>14s}: {iou:.4f}")


if __name__ == "__main__":
    main()