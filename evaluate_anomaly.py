"""evaluate_anomaly.py — Steps 6, 7 & 8: Out-of-Distribution Evaluation"""

import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets_ood import FishyscapesLostAndFound, SMIYCDataset
from data.transforms import get_val_transform_erfnet
from utils.anomaly_methods import (
    msp_anomaly_score, rba_anomaly_score,
    maxlogit_anomaly_score, maxentropy_anomaly_score,
)
from utils.metrics import evaluate_anomaly
from utils.logger import setup_logging
from utils.visualization import visualize_prediction

TEMPERATURE_SEARCH_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def load_model(model_name, ckpt_path, cfg, device, eomt_version="finetuned"):
    if model_name == "erfnet":
        from models.erfnet import ERFNet
        model = ERFNet(num_classes=20)
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=str(device), weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state" in state:
                state = state["model_state"]
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith("module.") else k
                new_state[name] = v
            model.load_state_dict(new_state, strict=False)
            print("[ERFNet] Pre-trained weights loaded successfully.")
        model._img_size = (512, 1024)
        return model.to(device).eval()

    if model_name != "eomt":
        raise ValueError(f"Unknown model: {model_name}")

    # Tutti gli EoMT (cityscapes, coco, finetuned) usano architettura ufficiale TU/e
    if eomt_version not in ("cityscapes", "coco", "finetuned"):
        raise ValueError(f"Unknown eomt_version: {eomt_version}")

    from models.eomt.eomt_orig.eomt import EoMT as EoMTOrig
    from models.eomt.eomt_orig.vit import ViT

    peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    peek = peek.get("state_dict", peek) if isinstance(peek, dict) else peek
    peek = {k.replace("network.", "", 1): v for k, v in peek.items()
            if not k.startswith("criterion")}
    n_patches = peek["encoder.backbone.pos_embed"].shape[1]
    grid_size = int(n_patches ** 0.5)
    img_size = (grid_size * 16, grid_size * 16)
    num_q = peek["q.weight"].shape[0]
    num_classes = peek["class_head.weight"].shape[0] - 1
    print(f"[EoMT-{eomt_version}] img_size={img_size}, num_q={num_q}, num_classes={num_classes}")

    vit = ViT(img_size=img_size, patch_size=16,
              backbone_name="vit_base_patch14_reg4_dinov2")
    model = EoMTOrig(encoder=vit, num_classes=num_classes,
                     num_q=num_q, num_blocks=3, masked_attn_enabled=True)
    missing, unexpected = model.load_state_dict(peek, strict=False)
    print(f"[EoMT-{eomt_version}] Loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model._is_orig = True
    model._img_size = img_size
    return model.to(device).eval()


def load_ood_dataset(dataset_name, cfg, model_name="eomt", image_size=None):
    # Tutti i modelli (ERFNet, EoMT-orig) usano range [0,1]
    transform = (get_val_transform_erfnet(image_size) if image_size
                 else get_val_transform_erfnet())

    if dataset_name == "fs_laf":
        return FishyscapesLostAndFound(root=cfg["dataset"]["fs_laf_root"],
                                       image_transform=transform)
    if dataset_name == "fs_static":
        return FishyscapesLostAndFound(root=cfg["dataset"]["fs_static_root"],
                                       image_transform=transform)
    if dataset_name == "smiyc_ra21":
        return SMIYCDataset(root=cfg["dataset"]["smiyc_root"], subset="RoadAnomaly21",
                            image_transform=transform)
    if dataset_name == "smiyc_ro21":
        return SMIYCDataset(root=cfg["dataset"]["smiyc_root"], subset="RoadObstacle21",
                            image_transform=transform)
    if dataset_name == "road_anomaly":
        # Struttura flat: images/ + labels_masks/ (Lis et al.)
        return FishyscapesLostAndFound(root=cfg["dataset"]["road_anomaly_root"],
                                       image_transform=transform)
    raise ValueError(f"Unknown OoD dataset: {dataset_name}")


def compute_scores(model, dataloader, device, model_name, method_name, cfg,
                   temperature, cache_dir, save_vis=False, vis_dir=None):
    from torch import autocast
    use_cache = cfg["logits_cache"]["use_cache"]
    all_scores, all_labels = [], []

    for idx, (images, labels) in enumerate(tqdm(dataloader, desc="Scoring")):
        cache_path = cache_dir / f"{idx:05d}.npy"
        eomt_logits_path = cache_dir / f"{idx:05d}_cls.npy"
        images = images.to(device)

        if model_name == "erfnet":
            if use_cache and cache_path.exists():
                logits = torch.from_numpy(np.load(str(cache_path)))
            else:
                with autocast(device_type=device.type):
                    logits = model(images).detach().cpu()
                np.save(str(cache_path), logits.numpy())
            if logits.shape[1] > 19:
                logits = logits[:, :19]
            if method_name == "msp":
                scores = msp_anomaly_score(logits, temperature=temperature)
            elif method_name == "maxlogit":
                scores = maxlogit_anomaly_score(logits, temperature=temperature)
            elif method_name == "maxentropy":
                scores = maxentropy_anomaly_score(logits, temperature=temperature)
        else:
            if use_cache and cache_path.exists() and eomt_logits_path.exists():
                pred_masks = torch.from_numpy(np.load(str(cache_path))).float()
                pred_logits = torch.from_numpy(np.load(str(eomt_logits_path))).float()
            else:
                with autocast(device_type=device.type):
                    out = model(images)
                mask_list, class_list = out
                pred_masks = mask_list[-1].float().detach().cpu()
                pred_logits = class_list[-1].float().detach().cpu()
                pred_masks = F.interpolate(pred_masks, size=images.shape[-2:],
                                           mode="bilinear", align_corners=False)
                np.save(str(cache_path), pred_masks.numpy())
                np.save(str(eomt_logits_path), pred_logits.numpy())

            if method_name == "rba":
                scores = rba_anomaly_score(pred_logits.float(), pred_masks.float(),
                                           temperature=temperature)
            else:
                mask_cls = torch.softmax(pred_logits[..., :-1].float() / temperature, dim=-1)
                mask_pred = torch.sigmoid(pred_masks.float())
                dense = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                if method_name == "msp":
                    scores = 1.0 - dense.max(dim=1).values
                elif method_name == "maxlogit":
                    scores = -dense.max(dim=1).values
                elif method_name == "maxentropy":
                    dense = dense / (dense.sum(dim=1, keepdim=True) + 1e-6)
                    scores = -(dense * torch.log(dense.clamp(min=1e-12))).sum(dim=1)

        scores_np = scores.detach().cpu()
        H_gt, W_gt = labels.shape[-2], labels.shape[-1]
        if scores_np.shape[-2:] != (H_gt, W_gt):
            scores_np = F.interpolate(scores_np.unsqueeze(1).float(), size=(H_gt, W_gt),
                                       mode="bilinear", align_corners=False).squeeze(1)

        if save_vis and vis_dir is not None and idx < 10:
            visualize_prediction(
                image=images.cpu(),
                gt_mask=labels[0].cpu().numpy(),
                pred_mask=labels[0].cpu().numpy(),
                anomaly_score=scores_np[0].cpu().numpy(),
                save_path=str(vis_dir / f"{idx:05d}.png"),
            )

        all_scores.append(scores_np.numpy().flatten())
        all_labels.append(labels.numpy().flatten())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def run_single(args, cfg, logger, device, temperatures):
    if args.model == "erfnet":
        ckpt_path = cfg["checkpoints"]["erfnet"]
    elif args.eomt_version == "finetuned":
        ckpt_path = cfg["checkpoints"]["eomt_finetuned"]
    elif args.eomt_version == "cityscapes":
        ckpt_path = cfg["checkpoints"]["eomt_cityscapes"]
    else:
        ckpt_path = cfg["checkpoints"]["eomt_coco"]

    model = load_model(args.model, ckpt_path, cfg, device, eomt_version=args.eomt_version)
    img_size = getattr(model, "_img_size", None)

    dataset = load_ood_dataset(args.dataset, cfg, model_name=args.model, image_size=img_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    cache_id = args.model if args.model == "erfnet" else f"eomt_{args.eomt_version}"
    cache_dir = Path(cfg["logits_cache"]["save_dir"]) / f"{cache_id}_{args.dataset}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    best_fpr95 = float("inf")
    best_T = temperatures[0]

    for T in temperatures:
        vis_dir = None
        if args.save_vis:
            vis_dir = Path("results/visualizations") / f"{cache_id}_{args.method}_{args.dataset}_T{T}"
            vis_dir.mkdir(parents=True, exist_ok=True)

        scores, labels = compute_scores(model, dataloader, device, args.model,
                                        args.method, cfg, T, cache_dir,
                                        save_vis=args.save_vis, vis_dir=vis_dir)
        results = evaluate_anomaly(scores, labels)
        logger.info(f"[{cache_id}][{args.method}][T={T:.2f}] "
                    f"AuPRC={results['AuPRC']:.4f} | FPR95={results['FPR95']:.4f} | "
                    f"AuROC={results['AuROC']:.4f}")
        if results["FPR95"] < best_fpr95:
            best_fpr95 = results["FPR95"]
            best_T = T

    if len(temperatures) > 1:
        logger.info(f"Best temperature: T={best_T:.2f} → FPR95={best_fpr95:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_eval.yaml")
    parser.add_argument("--method", choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
    parser.add_argument("--model", choices=["erfnet", "eomt"], default="erfnet")
    parser.add_argument("--eomt-version", choices=["cityscapes", "coco", "finetuned"], default="finetuned")
    parser.add_argument("--dataset", choices=["fs_laf", "fs_static", "smiyc_ra21", "smiyc_ro21", "road_anomaly"], default="fs_laf")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-search", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temperatures = TEMPERATURE_SEARCH_VALUES if args.temperature_search else [args.temperature]
    run_single(args, cfg, logger, device, temperatures)


if __name__ == "__main__":
    main()