import numpy as np
import torch


# Segmentation Metrics

class MeanIoUMeter:

    def __init__(self, num_classes: int = 19, ignore_index: int=255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred   = pred.cpu().numpy().flatten().astype(np.int64)
        target = target.cpu().numpy().flatten().astype(np.int64)

        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]

        valid_pred = pred != self.ignore_index
        pred, target = pred[valid_pred], target[valid_pred]
        
        pred = np.clip(pred, 0, self.num_classes - 1)
        target = np.clip(target, 0, self.num_classes - 1)

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


# Anomaly Detection Metrics

def compute_auprc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, anomaly_scores))


def compute_fpr95(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    idx = np.searchsorted(tpr, 0.95)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_auroc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, anomaly_scores))


def evaluate_anomaly(anomaly_scores, gt_masks):
    anomaly_scores = np.array(anomaly_scores).flatten()
    gt_masks = np.array(gt_masks).flatten()

    if anomaly_scores.shape != gt_masks.shape:
        print(f"[Warning] Mismatch! Scores: {anomaly_scores.shape}, GT: {gt_masks.shape}")
        min_len = min(anomaly_scores.shape[0], gt_masks.shape[0])
        anomaly_scores = anomaly_scores[:min_len]
        gt_masks = gt_masks[:min_len]

    mask = (gt_masks != 255) 
    
    scores_filtered = anomaly_scores[mask].astype(np.float32)
    gt_filtered = gt_masks[mask].astype(np.int32)

    return {
        "AuPRC": compute_auprc(scores_filtered, gt_filtered),
        "FPR95": compute_fpr95(scores_filtered, gt_filtered),
        "AuROC": compute_auroc(scores_filtered, gt_filtered),
    }
