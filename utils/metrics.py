"""
utils/metrics.py
Evaluation metrics for semantic segmentation and anomaly detection.
"""

import warnings
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


class MeanIoUMeter:
    """
    Streaming mIoU meter. Call update() each batch, compute() at the end.
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.cpu().numpy().flatten().astype(np.int64)
        target = target.cpu().numpy().flatten().astype(np.int64)

        # ignore void pixels
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]

        pred = np.clip(pred, 0, self.num_classes - 1)

        # warn if target has out-of-range values (e.g. bad COCO->Cityscapes mapping)
        out_of_range = int(np.sum((target < 0) | (target >= self.num_classes)))
        if out_of_range > 0:
            warnings.warn(
                f"{out_of_range} target pixels were out of range "
                f"[0, {self.num_classes - 1}] and have been clipped. "
                "Check your class mapping.",
                stacklevel=2,
            )
        target = np.clip(target, 0, self.num_classes - 1)

        indices = self.num_classes * target + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_mat += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion_mat.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-10)
        return {
            "mIoU": float(np.nanmean(iou)),
            "per_class_iou": iou.tolist(),
            "pixel_acc": float(tp.sum() / (cm.sum() + 1e-10)),
        }

    def reset(self):
        self.confusion_mat[:] = 0


def compute_auprc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under Precision-Recall Curve. Higher is better."""
    return float(average_precision_score(labels, anomaly_scores))


def compute_fpr95(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """False Positive Rate at TPR=95%. Lower is better."""
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    indices = np.where(tpr >= 0.95)[0]
    if len(indices) == 0:
        return 1.0
    return float(fpr[indices[0]])


def compute_auroc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    return float(roc_auc_score(labels, anomaly_scores))


def evaluate_anomaly(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    ignore_value: int = 255,
) -> dict:
    """
    Compute anomaly metrics ignoring void pixels.

    Args:
        anomaly_scores: per-pixel anomaly scores (higher = more anomalous)
        labels:         binary ground-truth (1=anomaly, 0=normal, 255=ignore)
    """
    mask = labels != ignore_value
    scores = anomaly_scores[mask].astype(np.float32)
    gt = labels[mask].astype(np.int32)

    unexpected = set(np.unique(gt)) - {0, 1}
    if unexpected:
        raise ValueError(
            f"Anomaly labels must be binary (0/1). "
            f"Unexpected values found: {sorted(unexpected)}"
        )

    return {
        "AuPRC": compute_auprc(scores, gt),
        "FPR95": compute_fpr95(scores, gt),
        "AuROC": compute_auroc(scores, gt),
    }
