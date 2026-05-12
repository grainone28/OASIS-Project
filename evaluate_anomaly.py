"""
evaluate_anomaly.py — Steps 6, 7 & 8: Out-of-Distribution Evaluation
======================================================================
Evaluates anomaly segmentation on Fishyscapes / SMIYC benchmarks.

Step 6 & 7 — ERFNet + MSP:
  python evaluate_anomaly.py --config configs/anomaly_eval.yaml \
                             --method msp --model erfnet \
                             --dataset fishyscapes

Step 8 — EoMT + RbA:
  python evaluate_anomaly.py --config configs/anomaly_eval.yaml \
                             --method rba --model eomt \
                             --dataset smiyc

Temperature scaling grid search (fast, uses cached logits):
  python evaluate_anomaly.py --config configs/anomaly_eval.yaml \
                             --method msp --model erfnet \
                             --dataset fishyscapes --temperature-search
"""

import argparse
import os
import yaml
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cityscapes import CityscapesDataset
from data.datasets_ood import FishyscapesLostAndFound, SMIYCDataset
from data.transforms import get_val_transform, get_mask_transform
from utils.anomaly_methods import msp_anomaly_score, rba_anomaly_score
from utils.metrics import evaluate_anomaly
from utils.logger import setup_logging


def load_model(model_name, ckpt_path, cfg, device):
    if model_name == "erfnet":
        from models.erfnet import ERFNet
        model = ERFNet(num_classes=cfg["dataset"].get("cityscapes_classes", 19))
        if ckpt_path:
            model.load_pretrained(ckpt_path, device=str(device))
    elif model_name == "eomt":
        from models.eomt import EoMT
        model = EoMT(num_classes=19)
        if ckpt_path:
            model.load_pretrained(ckpt_path, device=str(device))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device).eval()


def load_ood_dataset(dataset_name, cfg):
    transform = get_val_transform()
    if dataset_name == "fishyscapes":
        return FishyscapesLostAndFound(
            root=cfg["dataset"]["fishyscapes_root"],
            transform=transform,
        )
    elif dataset_name == "smiyc_anomaly":
        return SMIYCDataset(
            root=cfg["dataset"]["smiyc_root"],
            subset="RoadAnomaly21",
            transform=transform,
        )
    elif dataset_name == "smiyc_obstacle":
        return SMIYCDataset(
            root=cfg["dataset"]["smiyc_root"],
            subset="RoadObstacle21",
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown OoD dataset: {dataset_name}")


@torch.no_grad()
def run_inference_and_cache(model, dataloader, device, model_name, cache_dir: Path, use_amp=True):
    """
    Run model forward pass, cache raw logits as .npy files.
    Returns all logits stacked and labels.
    """
    from torch.cuda.amp import autocast
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_scores, all_labels = [], []

    for idx, (images, labels) in enumerate(tqdm(dataloader, desc="Inference")):
        cache_path = cache_dir / f"{idx:05d}.npy"

        if cache_path.exists():
            logits = torch.from_numpy(np.load(str(cache_path)))
        else:
            images = images.to(device)
            with autocast(enabled=use_amp):
                if model_name == "erfnet":
                    logits = model(images).cpu()
                else:  # eomt
                    out = model(images)
                    # For caching convenience, store pred_masks as the "logits"
                    logits = out["pred_masks"].cpu()

            np.save(str(cache_path), logits.numpy())

        all_labels.append(labels.numpy())

    return all_labels


def compute_scores(model, dataloader, device, model_name, cfg, temperature, cache_dir):
    """Run inference (or load cache) and compute anomaly scores."""
    from torch.cuda.amp import autocast
    use_cache = cfg["logits_cache"]["use_cache"]
    cache_dir  = Path(cfg["logits_cache"]["save_dir"])

    all_scores, all_labels = [], []

    for idx, (images, labels) in enumerate(tqdm(dataloader, desc="Scoring")):
        cache_path = cache_dir / f"{idx:05d}.npy"
        eomt_logits_path = cache_dir / f"{idx:05d}_cls.npy"

        images = images.to(device)

        if model_name == "erfnet":
            if use_cache and cache_path.exists():
                logits = torch.from_numpy(np.load(str(cache_path)))
            else:
                with autocast():
                    logits = model(images).cpu()
                np.save(str(cache_path), logits.numpy())

            scores = msp_anomaly_score(logits, temperature=temperature)  # [B, H, W]

        else:  # EoMT + RbA
            if use_cache and cache_path.exists() and eomt_logits_path.exists():
                pred_masks  = torch.from_numpy(np.load(str(cache_path)))
                pred_logits = torch.from_numpy(np.load(str(eomt_logits_path)))
            else:
                with autocast():
                    out = model(images)
                pred_masks  = out["pred_masks"].cpu()
                pred_logits = out["pred_logits"].cpu()
                np.save(str(cache_path), pred_masks.numpy())
                np.save(str(eomt_logits_path), pred_logits.numpy())

            scores = rba_anomaly_score(pred_logits, pred_masks, temperature=temperature)

        all_scores.append(scores.numpy().flatten())
        all_labels.append(labels.numpy().flatten())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/anomaly_eval.yaml")
    parser.add_argument("--method",  choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")    
    parser.add_argument("--model",    choices=["erfnet", "eomt"], default="erfnet")
    parser.add_argument("--dataset",  choices=["fishyscapes", "smiyc_anomaly", "smiyc_obstacle"],
                        default="fishyscapes")
    parser.add_argument("--temperature",        type=float, default=1.0)
    parser.add_argument("--temperature-search", action="store_true",
                        help="Run grid search over temperature values")
    args = parser.parse_args()

    logger = setup_logging()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_map = {
        "erfnet": cfg["checkpoints"]["erfnet"],
        "eomt":   cfg["checkpoints"]["eomt_finetuned"],
    }
    model = load_model(args.model, ckpt_map[args.model], cfg, device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset    = load_ood_dataset(args.dataset, cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    cache_dir = Path(cfg["logits_cache"]["save_dir"]) / f"{args.model}_{args.dataset}"

    # ── Temperature grid search ───────────────────────────────────────────────
    if args.temperature_search:
# Ensure we load the configuration for the correct method
        method_name = args.method.lower()
        ts_cfg = cfg["methods"][method_name]["temperature_scaling"]        
        temperatures = np.arange(ts_cfg["t_min"], ts_cfg["t_max"] + ts_cfg["t_step"],
                                 ts_cfg["t_step"])
        best_fpr95, best_T = float("inf"), 1.0

        for T in temperatures:
            scores, labels = compute_scores(model, dataloader, device, args.model,
                                            cfg, T, cache_dir)
            results = evaluate_anomaly(scores, labels)
            logger.info(f"T={T:.2f} | AuPRC={results['AuPRC']:.4f} | FPR95={results['FPR95']:.4f}")
            if results["FPR95"] < best_fpr95:
                best_fpr95, best_T = results["FPR95"], T

        logger.info(f"\n★ Best T={best_T:.2f}  FPR95={best_fpr95:.4f}")
        return

    # ── Single evaluation ─────────────────────────────────────────────────────
    scores, labels = compute_scores(model, dataloader, device, args.model,
                                    cfg, args.temperature, cache_dir)

    results = evaluate_anomaly(scores, labels)

    logger.info("=" * 50)
    logger.info(f"Model   : {args.model.upper()}")
    logger.info(f"Method  : {args.method.upper()}")
    logger.info(f"Dataset : {args.dataset}")
    logger.info(f"Temp T  : {args.temperature}")
    logger.info(f"AuPRC   : {results['AuPRC']:.4f}")
    logger.info(f"FPR95   : {results['FPR95']:.4f}")
    logger.info(f"AuROC   : {results['AuROC']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
