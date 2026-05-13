"""
models/lora.py
Low-Rank Adaptation (LoRA) for transformer attention layers.
Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

How it works:
  Original weight W ∈ R^{d×k} is frozen.
  Two trainable matrices A ∈ R^{r×k} and B ∈ R^{d×r} are injected.
  Forward: y = x @ W.T + (x @ A.T @ B.T) * (alpha / r)

Usage:
  from models.lora import inject_lora
  inject_lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a LoRA branch.

    Args:
        original: the nn.Linear layer to wrap
        rank:     inner dimension r
        alpha:    scaling factor (weight = alpha/r * B @ A)
        dropout:  applied to the input before the LoRA branch
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank     = rank
        self.scaling  = alpha / rank

        in_features  = original.in_features
        out_features = original.out_features

        # LoRA matrices — initialised so that BA = 0 at the start of training
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # B = 0 → output unchanged at init

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = self.original(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta

    def extra_repr(self) -> str:
        return (f"in={self.original.in_features}, "
                f"out={self.original.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}")


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ("qkv", "proj"),
) -> nn.Module:
    """
    Walk the model and replace all nn.Linear layers whose name matches
    any entry in `target_modules` with a LoRALinear wrapper.

    Returns the modified model (in-place modification).
    """
    replaced = 0
    for module_name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr_name in target_modules:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, attr_name, lora_layer)
                replaced += 1

    print(f"[LoRA] Injected into {replaced} linear layers "
          f"(rank={rank}, alpha={alpha}, target={list(target_modules)})")
    return model


def count_trainable_params(model: nn.Module) -> dict:
    """Report total vs trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }
