"""
utils/metrics.py
Evaluation metrics for semantic segmentation and anomaly detection.
"""

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Segmentation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class MeanIoUMeter:
    """
    Streaming mean Intersection-over-Union for semantic segmentation.

    Usage:
        meter = MeanIoUMeter(num_classes=19, ignore_index=255)
        for pred, target in dataloader:
            meter.update(pred, target)
        results = meter.compute()
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   [B, H, W] — predicted class indices (long tensor)
            target: [B, H, W] — ground-truth class indices (long tensor)
        """
        pred   = pred.cpu().numpy().flatten().astype(np.int64)
        target = target.cpu().numpy().flatten().astype(np.int64)

        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]

        # Clip to valid range to avoid index errors
        pred = np.clip(pred, 0, self.num_classes - 1)

        indices = self.num_classes * target + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_mat += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion_mat.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp
        fn  = cm.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-10)
        miou = np.nanmean(iou)
        return {
            "mIoU":    float(miou),
            "per_class_iou": iou.tolist(),
            "pixel_acc": float(tp.sum() / cm.sum()),
        }

    def reset(self):
        self.confusion_mat[:] = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Anomaly Detection Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_auprc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Area under the Precision-Recall Curve (primary anomaly metric).

    Args:
        anomaly_scores: [N] — higher = more anomalous
        labels:         [N] — binary (1 = anomaly, 0 = in-distribution)

    Returns:
        AuPRC in [0, 1]; random baseline ≈ fraction of positive pixels.
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, anomaly_scores))


def compute_fpr95(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    False Positive Rate at 95% True Positive Rate (FPR95).
    Lower is better (ideal = 0.0).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    # Find threshold where TPR ≥ 0.95
    idx = np.searchsorted(tpr, 0.95)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_auroc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the ROC Curve (supplementary metric)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, anomaly_scores))


def evaluate_anomaly(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    ignore_value: int = 255,
) -> dict:
    """
    Compute all anomaly metrics, ignoring void pixels.

    Args:
        anomaly_scores: flat array of per-pixel anomaly scores
        labels:         flat array of binary ground-truth (1=anomaly, 255=ignore)
    """
    mask = labels != ignore_value
    scores = anomaly_scores[mask].astype(np.float32)
    gt     = labels[mask].astype(np.int32)

    return {
        "AuPRC":  compute_auprc(scores, gt),
        "FPR95":  compute_fpr95(scores, gt),
        "AuROC":  compute_auroc(scores, gt),
    }
