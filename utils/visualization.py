"""
utils/visualization.py
Helpers for visualising segmentation predictions and anomaly maps.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import Optional

# Cityscapes colour palette (19 trainIds)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
], dtype=np.uint8)

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def mask_to_rgb(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Convert a [H, W] class-index mask to [H, W, 3] RGB using Cityscapes palette.
    Ignore pixels are rendered in white.
    """
    h, w = mask.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255  # default white

    for class_id, colour in enumerate(CITYSCAPES_PALETTE):
        rgb[mask == class_id] = colour

    return rgb


def visualize_prediction(
    image: torch.Tensor,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    anomaly_score: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Side-by-side visualisation: input | GT | prediction [| anomaly map].
    """
    # De-normalise image (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    n_cols = 4 if anomaly_score is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(gt_mask))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_to_rgb(pred_mask))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if anomaly_score is not None:
        im = axes[3].imshow(anomaly_score, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[3].set_title("Anomaly Score")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pr_curve(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
):
    """Plot PR curve for anomaly detection evaluation."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    mask = labels != 255
    scores_m = anomaly_scores[mask]
    labels_m = labels[mask]

    precision, recall, _ = precision_recall_curve(labels_m, scores_m)
    ap = average_precision_score(labels_m, scores_m)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
