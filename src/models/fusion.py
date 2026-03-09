"""
Gated Tanh Unit (GTU) fusion mechanism.

Inspired by *Tips and Tricks for VQA* (Teney et al., 2018).

::

    y = tanh(W₁·x + b₁)  ⊙  σ(W₂·x + b₂)        # element-wise gating
    output = LayerNorm(y + residual(x))               # residual + normalize
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedTanhFusion(nn.Module):
    """GTU fusion with residual connection and LayerNorm.

    Fuses an arbitrary-length concatenation of context vectors
    (e.g. ``[text_ctx, img_ctx, hidden]``) into a single ``output_dim``
    vector through learned gating.

    Args:
        input_dim:  Dimension of the concatenated input.
        output_dim: Desired output dimension.
        dropout:    Dropout rate applied after gating.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc_tanh = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection — identity when dims already match
        self.residual: nn.Module = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, input_dim) → (B, output_dim)``."""
        y_tanh = torch.tanh(self.fc_tanh(x))
        y_gate = torch.sigmoid(self.fc_gate(x))
        y = y_tanh * y_gate             # element-wise gating
        y = y + self.residual(x)        # residual shortcut
        y = self.layer_norm(y)
        y = self.dropout(y)
        return y
