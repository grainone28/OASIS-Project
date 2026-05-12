"""
utils/anomaly_methods.py
Post-hoc anomaly scoring methods:
  - MSP  (Maximum Softmax Probability)   — for pixel-based models like ERFNet
  - RbA  (Rejected by All)               — for mask-based models like EoMT
  - Temperature Scaling calibration
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Maximum Softmax Probability (Steps 6 & 7 — ERFNet)
# ═══════════════════════════════════════════════════════════════════════════════

def msp_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute the per-pixel MSP anomaly score from raw logits.

    A HIGH anomaly score → likely OoD.
    Score = 1 - max_class(softmax(logits / T))

    Args:
        logits:      [B, C, H, W] — raw (pre-softmax) class scores
        temperature: scaling factor T (T > 1 softens distribution)

    Returns:
        anomaly_score: [B, H, W] in [0, 1]
    """
    probs = F.softmax(logits / temperature, dim=1)  # [B, C, H, W]
    max_prob = probs.max(dim=1).values               # [B, H, W]
    return 1.0 - max_prob                            # high → uncertain → anomaly


def msp_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    """Load cached logits from disk and compute MSP scores."""
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = msp_anomaly_score(logits, temperature=temperature)
    return scores.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Rejected by All — RbA (Step 8 — EoMT)
# ═══════════════════════════════════════════════════════════════════════════════

def rba_anomaly_score(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    RbA: Rejected by All anomaly score for mask-based architectures.

    Core idea: a pixel is anomalous if NO query "claims" it with high confidence.

    Algorithm:
      1. Compute per-query mask sigmoid probabilities → how strongly each
         query claims each pixel.
      2. Compute per-query class confidence (max over known classes, excluding
         the "no-object" logit at index -1).
      3. Per-pixel score = weighted sum of mask probs, weighted by class conf.
      4. Anomaly score = 1 − max_{query} (class_conf_q × mask_prob_q(pixel))

    Args:
        pred_logits: [B, Q, C+1]  — class logits per query (last = no-object)
        pred_masks:  [B, Q, H, W] — mask logits per query per pixel
        temperature: T for calibration of class logits

    Returns:
        anomaly_score: [B, H, W] in [0, 1]
    """
    # Class confidence: softmax over known classes (exclude no-object column)
    known_logits = pred_logits[..., :-1] / temperature       # [B, Q, C]
    class_conf   = F.softmax(known_logits, dim=-1).max(dim=-1).values  # [B, Q]

    # Mask probability: how strongly each query claims each pixel
    mask_prob = torch.sigmoid(pred_masks)                    # [B, Q, H, W]

    # Per-pixel confidence: max over queries of (class_conf × mask_prob)
    class_conf_expanded = class_conf.unsqueeze(-1).unsqueeze(-1)  # [B, Q, 1, 1]
    query_score = class_conf_expanded * mask_prob            # [B, Q, H, W]
    max_query_score = query_score.max(dim=1).values          # [B, H, W]

    # Anomaly = "rejected by all known queries"
    return 1.0 - max_query_score                             # [B, H, W]




# ═══════════════════════════════════════════════════════════════════════════════
# MaxLogit & MaxEntropy (Step 8 — Metodi Post-Hoc)
# ═══════════════════════════════════════════════════════════════════════════════

def maxlogit_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Calcola l'anomaly score basato sul Maximum Logit.
    Score = - max(logits / T)
    Valori più alti indicano maggiore probabilità di anomalia.
    """
    scaled_logits = logits / temperature
    max_logits = scaled_logits.max(dim=1).values
    return -max_logits

def maxentropy_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Calcola l'anomaly score basato sull'Entropia di Shannon.
    Score = - sum(p * log(p))
    Maggiore è l'entropia, più il modello è incerto (anomalia).
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    # Aggiungo 1e-12 per evitare logaritmo di zero (NaN)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return entropy

def maxlogit_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    """Load cached logits from disk and compute MaxLogit scores."""
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = maxlogit_anomaly_score(logits, temperature=temperature)
    return scores.numpy()

def maxentropy_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    """Load cached logits from disk and compute MaxEntropy scores."""
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = maxentropy_anomaly_score(logits, temperature=temperature)
    return scores.numpy()



# ═══════════════════════════════════════════════════════════════════════════════
# Temperature Scaling Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search_temperature(
    logits_dir: str,
    labels: np.ndarray,
    method: str = "msp",           # "msp" | "rba"
    t_min: float = 0.5,
    t_max: float = 2.0,
    t_step: float = 0.05,
    metric: str = "AuPRC",
    ignore_value: int = 255,
) -> Dict[str, float]:
    """
    Efficiently sweep temperature values over pre-saved logits to find
    the best calibration without re-running the model.

    Args:
        logits_dir: directory containing .npy or .pt logit files
        labels:     flat ground-truth array (1=anomaly, 255=ignore)
        method:     scoring method to calibrate
        metric:     metric to maximise ("AuPRC") or minimise ("FPR95")

    Returns:
        {"best_T": float, "best_score": float}
    """
    import os
    from utils.metrics import evaluate_anomaly

    temps   = np.arange(t_min, t_max + t_step, t_step)
    results = {}

    # Load all logit files once
    logit_files = sorted([
        os.path.join(logits_dir, f)
        for f in os.listdir(logits_dir)
        if f.endswith(".npy") or f.endswith(".pt")
    ])

    for T in temps:
        all_scores = []
        for path in logit_files:
            if method == "msp":
                scores = msp_from_logits_file(path, temperature=T)
            elif method == "maxlogit":
                scores = maxlogit_from_logits_file(path, temperature=T)
            elif method == "maxentropy":
                scores = maxentropy_from_logits_file(path, temperature=T)
            elif method == "rba":
                raise NotImplementedError("Caching for RbA requires both masks and logits, use the loop in evaluate_anomaly.py instead of this function.")
            else:
                raise ValueError(f"Method {method} not recognized.")
            
            all_scores.append(scores.flatten())

        eval_result = evaluate_anomaly(all_scores, labels, ignore_value=ignore_value)
        results[round(float(T), 4)] = eval_result[metric]

    if metric == "FPR95":
        best_T = min(results, key=results.__getitem__)
    else:
        best_T = max(results, key=results.__getitem__)

    print(f"[TempScaling] Best T={best_T:.3f}  {metric}={results[best_T]:.4f}")
    return {"best_T": best_T, "best_score": results[best_T], "all_results": results}
