"""
train.py — Step 5: Parameter-Efficient Fine-Tuning of EoMT
===========================================================
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cityscapes import CityscapesDataset
from data.transforms import get_train_transform, get_val_transform, get_mask_transform
from models.eomt import build_eomt
from models.lora import inject_lora, count_trainable_params
from utils.metrics import MeanIoUMeter
from utils.logger import setup_logging, MetricLogger
from utils.postprocessing import queries_to_segmentation_fast


# ──────────────────────────────────────────────────────────
# Optimizer / Scheduler
# ──────────────────────────────────────────────────────────
def build_optimizer(model, cfg):
    lr = cfg["train"]["learning_rate"]
    wd = cfg["train"]["weight_decay"]
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    warmup_steps = cfg["train"]["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["train"]["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────
# FIXED PANOPTIC LOSS (CORRETTA CON NORMALIZZAZIONE)
# ──────────────────────────────────────────────────────────
def panoptic_loss(outputs, targets, num_classes, ignore_index=255):
    pred_masks  = outputs["pred_masks"]   # [B, Q, H, W]
    pred_logits = outputs["pred_logits"]  # [B, Q, C+1]

    mask_cls = torch.softmax(pred_logits, dim=-1)[..., :-1]  # [B, Q, C]
    mask_pred = torch.sigmoid(pred_masks)                    # [B, Q, H, W]

    dense_probs = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

    # 🔥 normalizzazione fondamentale
    dense_probs = dense_probs / (dense_probs.sum(dim=1, keepdim=True) + 1e-6)

    log_probs = torch.log(dense_probs.clamp(min=1e-6))

    return nn.functional.nll_loss(
        log_probs, targets, ignore_index=ignore_index
    )


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    num_classes = cfg["dataset"]["num_classes"]
    accum_steps = cfg["train"]["gradient_accumulation_steps"]
    use_amp     = cfg["train"]["amp"]

    optimizer.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    for step, (images, targets) in pbar:
        images  = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = panoptic_loss(outputs, targets, num_classes)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if cfg["train"]["clip_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["clip_grad_norm"]
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps

        pbar.set_postfix(
            loss=f"{loss.item() * accum_steps:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

        if step == 0 and epoch == 1:
            print("DEBUG logits sample:",
                  outputs["pred_logits"][0, :5].detach().cpu())

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, num_classes, use_amp=True):
    model.eval()
    meter = MeanIoUMeter(num_classes=num_classes)

    for images, targets in tqdm(loader, desc="Validating"):
        images  = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)

        pred_masks_sig = torch.sigmoid(outputs["pred_masks"])
        pred_logits_cls = outputs["pred_logits"][..., :-1]

        preds = queries_to_segmentation_fast(
            pred_masks=pred_masks_sig,
            pred_logits=pred_logits_cls,
            threshold=0.5
        )

        meter.update(preds, targets)

    return meter.compute()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_eomt.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg["paths"]["log_dir"], cfg["logging"]["run_name"])
    metric_logger = MetricLogger(
        save_dir=cfg["paths"]["log_dir"],
        run_name=cfg["logging"]["run_name"],
        use_wandb=cfg["logging"]["use_wandb"],
        wandb_project=cfg["logging"]["project"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Model
    model = build_eomt(cfg)

    # 🔴 CHECK class mismatch
    if hasattr(model, "class_embed"):
        w = model.class_embed.weight
        num_ckpt_classes = w.shape[0]
        expected = cfg["dataset"]["num_classes"] + 1

        logger.info(f"[CHECK] checkpoint classes: {num_ckpt_classes}")

        if num_ckpt_classes != expected:
            raise ValueError(
                f"Class mismatch: checkpoint={num_ckpt_classes}, expected={expected}"
            )

    # LoRA
    if cfg["lora"]["enabled"]:
        model = inject_lora(
            model,
            rank=cfg["lora"]["rank"],
            alpha=cfg["lora"]["alpha"],
            dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
        )

    param_info = count_trainable_params(model)
    logger.info(
        f"Parameters — Total: {param_info['total']:,}  "
        f"Trainable: {param_info['trainable']:,}  "
        f"({param_info['trainable_pct']:.1f}%)"
    )

    model = model.to(device)

    # Data
    img_size = tuple(cfg["dataset"]["image_size"])

    train_ds = CityscapesDataset(
        cfg["dataset"]["root"], "train",
        transform=get_train_transform(img_size),
        target_transform=get_mask_transform(img_size),
    )

    val_ds = CityscapesDataset(
        cfg["dataset"]["root"], "val",
        transform=get_val_transform(img_size),
        target_transform=get_mask_transform(img_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler    = GradScaler(enabled=cfg["train"]["amp"])

    best_miou = 0.0
    ckpt_dir  = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch
        )

        val_results = validate(
            model,
            val_loader,
            device,
            cfg["dataset"]["num_classes"],
            cfg["train"]["amp"]
        )

        miou = val_results["mIoU"]

        logger.info(
            f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | mIoU: {miou:.4f}"
        )

        metric_logger.log(epoch, {
            "train_loss": train_loss,
            "val_mIoU": miou
        })

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "mIoU": miou
            }, ckpt_dir / "eomt_finetuned_best.pth")

            logger.info(f"↑ New best mIoU: {best_miou:.4f}")

        if epoch % cfg["train"]["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict()
            }, ckpt_dir / f"eomt_epoch{epoch:03d}.pth")

    logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
    metric_logger.finish()


if __name__ == "__main__":
    main()