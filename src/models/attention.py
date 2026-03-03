"""Attention Mechanisms for VQA."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys))).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

class SpatialAttention(nn.Module):
    """Computes attention weights over image spatial regions."""
    def __init__(self, hidden_size: int, image_feat_dim: int = 512) -> None:
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_i = nn.Linear(image_feat_dim, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.W_q(query).unsqueeze(1)  # (B, 1, H)
        i_proj = self.W_i(image_features)      # (B, 49, H)
        scores = self.V(torch.tanh(q_proj + i_proj)).squeeze(2)  # (B, 49)
        weights = torch.softmax(scores, dim=1)                   # (B, 49)
        context = torch.bmm(weights.unsqueeze(1), image_features).squeeze(1) # (B, 512)
        return context, weights