"""
models/erfnet.py
ERFNet — Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation.
Reference: Romera et al., IEEE T-ITS 2018.

Architecture:
  Encoder: Downsampling blocks + non-bottleneck-1D residual blocks
  Decoder: Upsampling blocks + non-bottleneck-1D residual blocks
  Output:  per-pixel class logits [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class DownsamplingBlock(nn.Module):
    """Combines max-pool and a parallel convolution branch, then concatenates."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch - in_ch, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class NonBottleneck1D(nn.Module):
    """
    Factorized residual block using asymmetric (1×k, k×1) convolutions.
    Replaces a single 3×3 conv with two cheaper 1D convolutions.
    Supports dilated convolutions for enlarged receptive field.
    """

    def __init__(self, channels: int, dropout_p: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False)
        self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv3x1_2 = nn.Conv2d(
            channels, channels, (3, 1),
            padding=(dilation, 0), dilation=(dilation, 1), bias=False
        )
        self.conv1x3_2 = nn.Conv2d(
            channels, channels, (1, 3),
            padding=(0, dilation), dilation=(1, dilation), bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv3x1_1(x))
        x = F.relu(self.bn1(self.conv1x3_1(x)))
        x = F.relu(self.conv3x1_2(x))
        x = self.bn2(self.conv1x3_2(x))
        x = self.dropout(x)
        return F.relu(x + residual)


class UpsamplingBlock(nn.Module):
    """Transposed convolution for 2× upsampling."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# ── Full ERFNet ───────────────────────────────────────────────────────────────

class ERFNet(nn.Module):
    """
    ERFNet for semantic segmentation.

    Args:
        num_classes: number of output segmentation classes (19 for Cityscapes).
        dropout_p:   dropout probability inside non-bottleneck blocks.
    """

    def __init__(self, num_classes: int = 19, dropout_p: float = 0.3):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Stage 1: 3 → 16, 16 → 64
            DownsamplingBlock(3, 16),
            DownsamplingBlock(16, 64),
            NonBottleneck1D(64, dropout_p=0.03),
            NonBottleneck1D(64, dropout_p=0.03),
            NonBottleneck1D(64, dropout_p=0.03),
            NonBottleneck1D(64, dropout_p=0.03),
            NonBottleneck1D(64, dropout_p=0.03),

            # Stage 2: 64 → 128, dilated blocks
            DownsamplingBlock(64, 128),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=2),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=4),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=8),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=16),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=2),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=4),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=8),
            NonBottleneck1D(128, dropout_p=dropout_p, dilation=16),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            UpsamplingBlock(128, 64),
            NonBottleneck1D(64),
            NonBottleneck1D(64),

            UpsamplingBlock(64, 16),
            NonBottleneck1D(16),
            NonBottleneck1D(16),
        )

        # ── Output projection ─────────────────────────────────────────────────
        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalised RGB image
        Returns:
            logits: [B, num_classes, H, W] — raw (pre-softmax) scores
        """
        features = self.encoder(x)
        features = self.decoder(features)
        return self.output_conv(features)

    def load_pretrained(self, path: str, device: str = "cpu"):
        """Helper to load a .pth checkpoint."""
        state = torch.load(path, map_location=device)
        # Support both raw state-dict and wrapped checkpoints
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state)
        print(f"[ERFNet] Loaded weights from {path}")
