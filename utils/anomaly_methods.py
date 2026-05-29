"""
utils/anomaly_methods.py
Post-hoc anomaly scoring methods:
  - MSP  (Maximum Softmax Probability)   — for pixel-based models like ERFNet
  - MaxLogit                              — for pixel-based models
  - MaxEntropy (Shannon)                  — for pixel-based models
  - RbA  (Rejected by All)                — for mask-based models like EoMT
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
    """
    logits = logits.float()  # forza fp32 per stabilità numerica (autocast → fp16 → NaN)
    probs = F.softmax(logits / temperature, dim=1)
    max_prob = probs.max(dim=1).values
    return 1.0 - max_prob


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
    """
    pred_logits = pred_logits.float()
    pred_masks  = pred_masks.float()

    known_logits = pred_logits[..., :-1] / temperature       # [B, Q, C]
    class_conf   = F.softmax(known_logits, dim=-1).max(dim=-1).values  # [B, Q]

    mask_prob = torch.sigmoid(pred_masks)                    # [B, Q, H, W]

    class_conf_expanded = class_conf.unsqueeze(-1).unsqueeze(-1)
    query_score = class_conf_expanded * mask_prob
    max_query_score = query_score.max(dim=1).values

    return 1.0 - max_query_score


# ═══════════════════════════════════════════════════════════════════════════════
# MaxLogit & MaxEntropy (Step 8 — Metodi Post-Hoc)
# ═══════════════════════════════════════════════════════════════════════════════

def maxlogit_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Calcola l'anomaly score basato sul Maximum Logit.
    Score = - max(logits / T)
    Valori più alti indicano maggiore probabilità di anomalia.
    """
    logits = logits.float()  # forza fp32
    scaled_logits = logits / temperature
    max_logits = scaled_logits.max(dim=1).values
    return -max_logits


def maxentropy_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Calcola l'anomaly score basato sull'Entropia di Shannon.
    Score = - sum(p * log(p))
    Maggiore è l'entropia, più il modello è incerto (anomalia).

    Implementazione numericamente stabile:
      - fp32 forzato (evita NaN da autocast fp16)
      - log_softmax + exp invece di softmax + log (più stabile)
      - torch.nan_to_num finale come rete di sicurezza
    """
    logits = logits.float()  # forza fp32: con fp16 + logit estremi, softmax → 0/inf → NaN
    scaled = logits / temperature
    log_probs = F.log_softmax(scaled, dim=1)     # [B, C, H, W] — stabile anche per logit grandi
    probs     = log_probs.exp()                  # softmax stabile
    entropy   = -(probs * log_probs).sum(dim=1)  # [B, H, W]
    # Rete di sicurezza: sostituisci eventuali NaN/Inf residui con 0
    return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


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
    method: str = "msp",
    t_min: float = 0.5,
    t_max: float = 2.0,
    t_step: float = 0.05,
    metric: str = "AuPRC",
    ignore_value: int = 255,
) -> Dict[str, float]:
    """
    Efficiently sweep temperature values over pre-saved logits to find
    the best calibration without re-running the model.
    """
    import os
    from utils.metrics import evaluate_anomaly

    temps   = np.arange(t_min, t_max + t_step, t_step)
    results = {}

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