"""
utils/metrics.py
Evaluation metrics for semantic segmentation and anomaly detection.

──────────────────────────────────────────────────────────────────────────────
MODIFICHE (Step 4)
──────────────────────────────────────────────────────────────────────────────
Una sola modifica chirurgica in `MeanIoUMeter.update`:

  Prima, dopo `pred = pred[mask]`, c'era un singolo
      pred = np.clip(pred, 0, self.num_classes - 1)
  che schiacciava qualunque pred == ignore_index (255) sulla classe
  `num_classes - 1` (= 18, "bicycle"), gonfiando i falsi positivi di quella
  classe. Effetto invisibile finché tutte le predizioni sono in [0, 18],
  ma rompe il calcolo non appena si introduce un remap che produce 255
  (caso COCO → Cityscapes per le classi senza equivalente).

  Ora i pixel con pred == ignore_index vengono esclusi dalla confusion
  matrix prima del clip — convenzione standard di MMSegmentation. Vedi
  il docstring di `update` per i dettagli.

Il resto del file (`compute_auprc`, `compute_fpr95`, `compute_auroc`,
`evaluate_anomaly`) non è stato toccato.
──────────────────────────────────────────────────────────────────────────────
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

    def __init__(self, num_classes: int = 19, ignore_index: int=255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   [B, H, W] — predicted class indices (long tensor)
            target: [B, H, W] — ground-truth class indices (long tensor)

        Convenzione su `ignore_index`:
          - target == ignore_index → pixel completamente escluso (void GT).
          - pred  == ignore_index → pixel escluso anch'esso (predizione
            "non-classe", tipica dopo un remap COCO→Cityscapes per classi
            COCO che non hanno equivalente Cityscapes). È la convenzione
            standard di MMSegmentation: un modello che produce ignore non
            viene né premiato né penalizzato esplicitamente. Per il modello
            COCO questo significa che la mIoU è calcolata sulla porzione
            di pixel effettivamente predicibili nel suo spazio classi.
        """
        pred   = pred.cpu().numpy().flatten().astype(np.int64)
        target = target.cpu().numpy().flatten().astype(np.int64)

        # Escludi pixel con GT void.
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]

        # ── MODIFICA Step 4 ────────────────────────────────────────────────
        # Escludi pixel con pred = ignore_index (introdotti dal remap
        # COCO→Cityscapes). Senza questa riga, il np.clip sottostante li
        # mapperebbe a `num_classes - 1` falsificando la confusion matrix.
        valid_pred = pred != self.ignore_index
        pred, target = pred[valid_pred], target[valid_pred]
        # ───────────────────────────────────────────────────────────────────

        # Clamp difensivo per eventuali id fuori range residui.
        # Clamp difensivo per eventuali id fuori range residui.
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


def evaluate_anomaly(anomaly_scores, gt_masks):
    # 1. Assicuriamoci che siano array numpy flat
    anomaly_scores = np.array(anomaly_scores).flatten()
    gt_masks = np.array(gt_masks).flatten()

    # 2. RISOLUZIONE: Mismatch dimensionale (se le score e le GT non coincidono)
    if anomaly_scores.shape != gt_masks.shape:
        print(f"[Warning] Mismatch! Scores: {anomaly_scores.shape}, GT: {gt_masks.shape}")
        min_len = min(anomaly_scores.shape[0], gt_masks.shape[0])
        anomaly_scores = anomaly_scores[:min_len]
        gt_masks = gt_masks[:min_len]

    # 3. Creiamo la maschera per ignorare i pixel (255 è il valore standard di ignore)
    mask = (gt_masks != 255) 
    
    # 4. Applichiamo la maschera a entrambi
    # Nota: usiamo gt_masks (che è la variabile definita all'inizio della funzione)
    scores_filtered = anomaly_scores[mask].astype(np.float32)
    gt_filtered = gt_masks[mask].astype(np.int32)

    # 5. Calcolo delle metriche
    return {
        "AuPRC": compute_auprc(scores_filtered, gt_filtered),
        "FPR95": compute_fpr95(scores_filtered, gt_filtered),
        "AuROC": compute_auroc(scores_filtered, gt_filtered),
    }
