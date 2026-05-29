from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# Maximum Softmax Probability 

def msp_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = logits.float()  
    probs = F.softmax(logits / temperature, dim=1)
    max_prob = probs.max(dim=1).values
    return 1.0 - max_prob


def msp_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = msp_anomaly_score(logits, temperature=temperature)
    return scores.numpy()


# Rejected by All — RbA 

def rba_anomaly_score(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    
    pred_logits = pred_logits.float()
    pred_masks  = pred_masks.float()

    known_logits = pred_logits[..., :-1] / temperature      
    class_conf   = F.softmax(known_logits, dim=-1).max(dim=-1).values  

    mask_prob = torch.sigmoid(pred_masks)                    

    class_conf_expanded = class_conf.unsqueeze(-1).unsqueeze(-1)
    query_score = class_conf_expanded * mask_prob
    max_query_score = query_score.max(dim=1).values

    return 1.0 - max_query_score


# MaxLogit & MaxEntropy 

def maxlogit_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:

    logits = logits.float()  
    scaled_logits = logits / temperature
    max_logits = scaled_logits.max(dim=1).values
    return -max_logits


def maxentropy_anomaly_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    logits = logits.float()  
    scaled = logits / temperature
    log_probs = F.log_softmax(scaled, dim=1)     
    probs     = log_probs.exp()                  
    entropy   = -(probs * log_probs).sum(dim=1)  
    return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def maxlogit_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = maxlogit_anomaly_score(logits, temperature=temperature)
    return scores.numpy()


def maxentropy_from_logits_file(path: str, temperature: float = 1.0) -> np.ndarray:
    if path.endswith(".npy"):
        logits = torch.from_numpy(np.load(path))
    else:
        logits = torch.load(path, map_location="cpu")
    scores = maxentropy_anomaly_score(logits, temperature=temperature)
    return scores.numpy()


# Temperature Scaling Grid Search

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