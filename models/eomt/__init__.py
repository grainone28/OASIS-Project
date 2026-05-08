"""
models/eomt/__init__.py
Encoder-Only Mask Transformer (EoMT) wrapper.
Reference: "Your ViT is Secretly an Image Segmentation Model" (CVPR 2025).

EoMT eliminates the traditional Mask2Former decoder by concatenating
learnable query tokens directly to the image patch sequence inside DINOv2.
Self-attention in every ViT block lets queries interact with image features,
generating mask embeddings without any cross-attention decoder.

This module provides:
  - EoMT: the main model class
  - build_eomt: factory function to construct from a config dict
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class EoMT(nn.Module):
    """
    Encoder-Only Mask Transformer.

    Architecture summary:
      1. DINOv2 ViT backbone (patch size 14) produces patch tokens.
      2. `num_queries` learnable tokens are prepended to the token sequence.
      3. All tokens pass through every ViT self-attention block.
      4. The final query tokens are used as mask embeddings:
           mask_logits = query_embeddings @ patch_embeddings.T  → [B, Q, H/14, W/14]
      5. A linear classifier head assigns a class score to each query.

    Args:
        backbone_name:  timm model name, e.g. "vit_base_patch14_dinov2"
        num_queries:    number of learnable query tokens
        num_classes:    segmentation output classes (e.g. 19 for Cityscapes)
        freeze_layers:  freeze the first N ViT blocks (0 = no freezing)
        image_size:     (H, W) — must be divisible by patch_size (14)
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        num_queries: int = 100,
        num_classes: int = 19,
        freeze_layers: int = 9,
        image_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm>=0.9.16")

        self.num_queries  = num_queries
        self.num_classes  = num_classes
        self.image_size   = image_size

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=image_size,
            num_classes=0,   # remove classification head
        )
        self.embed_dim = self.backbone.embed_dim

        # ── Learnable query tokens ────────────────────────────────────────────
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, self.embed_dim) * 0.02
        )

        # ── Heads ─────────────────────────────────────────────────────────────
        # Class prediction: one score per query per class (+1 for "no-object")
        self.class_embed = nn.Linear(self.embed_dim, num_classes + 1)

        # Mask MLP: project query embedding before dot-product with patch tokens
        self.mask_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # ── Selective freezing ────────────────────────────────────────────────
        self._freeze_backbone_layers(freeze_layers)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _freeze_backbone_layers(self, n: int):
        """Freeze the first n transformer blocks of the ViT backbone."""
        if n <= 0:
            return
        frozen = 0
        for block in self.backbone.blocks[:n]:
            for param in block.parameters():
                param.requires_grad = False
            frozen += 1
        # Also freeze patch embedding and positional embedding
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        print(f"[EoMT] Froze {frozen} backbone blocks + patch embedding.")

    def _get_patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Run the backbone and return patch token feature map.
        Returns:
            patch_tokens: [B, N_patches, D]
            h, w: grid dimensions (H/14, W/14)
        """
        B = x.shape[0]
        # Embed patches
        x = self.backbone.patch_embed(x)
        # Add positional embedding (skip CLS token position)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed[:, 1:, :])

        # Prepend query tokens
        queries = self.query_tokens.expand(B, -1, -1)
        x = torch.cat([queries, x], dim=1)   # [B, Q + N_patches, D]

        # Forward through ViT blocks
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.norm(x)

        # Split back
        query_out = x[:, :self.num_queries, :]       # [B, Q, D]
        patch_out  = x[:, self.num_queries:, :]       # [B, N, D]

        # Derive spatial grid size
        h = self.image_size[0] // self.backbone.patch_embed.patch_size[0]
        w = self.image_size[1] // self.backbone.patch_embed.patch_size[1]

        return query_out, patch_out, h, w

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] normalised images

        Returns dict with:
            "pred_logits": [B, Q, num_classes+1]  — class scores per query
            "pred_masks":  [B, Q, H, W]            — upsampled binary masks
        """
        B, _, H, W = x.shape
        query_out, patch_out, h, w = self._get_patch_tokens(x)

        # Class predictions
        pred_logits = self.class_embed(query_out)   # [B, Q, C+1]

        # Mask predictions via dot product
        mask_emb   = self.mask_embed(query_out)     # [B, Q, D]
        patch_map  = patch_out.view(B, h, w, -1).permute(0, 3, 1, 2)  # [B, D, h, w]
        patch_flat = patch_map.view(B, -1, h * w)   # [B, D, h*w]

        # [B, Q, D] × [B, D, h*w] → [B, Q, h*w] → [B, Q, h, w]
        pred_masks = torch.bmm(mask_emb, patch_flat).view(B, self.num_queries, h, w)

        # Upsample masks to input resolution
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode="bilinear", align_corners=False)

        return {"pred_logits": pred_logits, "pred_masks": pred_masks}

    def load_pretrained(self, path: str, device: str = "cpu", strict: bool = True):
        state = torch.load(path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state, strict=strict)
        print(f"[EoMT] Loaded weights from {path}")


def build_eomt(cfg: dict) -> EoMT:
    """Construct EoMT from a config dict (parsed from YAML)."""
    return EoMT(
        backbone_name  = cfg["model"].get("backbone", "vit_base_patch14_dinov2"),
        num_queries    = cfg["model"].get("num_queries", 100),
        num_classes    = cfg["dataset"]["num_classes"],
        freeze_layers  = cfg["model"].get("freeze_backbone_layers", 9),
        image_size     = tuple(cfg["dataset"]["image_size"]),
    )
