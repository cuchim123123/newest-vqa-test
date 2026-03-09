"""Answer decoder with Spatial Attention + Bahdanau text attention + GTU fusion."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from src.data.dataset import PAD_IDX
from src.models.attention import BahdanauAttention, SpatialAttention
from src.models.encoder import CNNEncoder
from src.models.fusion import GatedTanhFusion

class AnswerDecoder(nn.Module):
    """Autoregressive answer decoder.

    At each time-step:
        1. Embed current token.
        2. Compute text attention context (Bahdanau) and image spatial context.
        3. Fuse ``[emb, text_ctx, img_ctx]`` through a **Gated Tanh Unit**.
        4. Feed fused vector into a 2-layer LSTM with residual + LayerNorm.
        5. Project to vocabulary logits.
    """

    def __init__(
        self, vocab_size: int, embed_size: int = 300, hidden_size: int = 512,
        num_layers: int = 2, dropout: float = 0.3, use_attention: bool = False,
        pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        if use_attention:
            self.text_attention = BahdanauAttention(hidden_size)
            self.spatial_attention = SpatialAttention(hidden_size, CNNEncoder.CNN_OUT_DIM)

        # ── GTU Fusion ──────────────────────────────────────────────
        # Input: [emb(embed_size) | text_ctx(hidden_size) | img_ctx(CNN_OUT_DIM)]
        fusion_in = embed_size + hidden_size + CNNEncoder.CNN_OUT_DIM
        self.fusion = GatedTanhFusion(fusion_in, hidden_size, dropout)

        # LSTM now receives hidden_size (from fusion) instead of raw concat
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.res_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
        img_feat: torch.Tensor, q_outputs: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(token.unsqueeze(1))  # (B, 1, E)

        if self.use_attention and q_outputs is not None:
            text_ctx, _ = self.text_attention(hidden[-1], q_outputs, q_mask)
            img_ctx, _ = self.spatial_attention(hidden[-1], img_feat)
            raw_concat = torch.cat([emb.squeeze(1), text_ctx, img_ctx], dim=-1)
        else:
            # No attention: use mean-pooled question outputs as text context
            # and mean-pooled image features as image context
            if q_outputs is not None:
                text_ctx = q_outputs.mean(dim=1)  # (B, H)
            else:
                text_ctx = hidden[-1]  # fallback
            img_ctx = img_feat.mean(dim=1)  # (B, CNN_OUT_DIM)
            raw_concat = torch.cat([emb.squeeze(1), text_ctx, img_ctx], dim=-1)

        # GTU fusion: (B, fusion_in) → (B, hidden_size)
        fused = self.fusion(raw_concat).unsqueeze(1)  # (B, 1, H)

        out, (hidden, cell) = self.lstm(fused, (hidden, cell))

        # Residual shortcut + LayerNorm
        residual = self.res_proj(fused.squeeze(1))
        out_res = self.layer_norm(out.squeeze(1) + residual)
        pred = self.fc(self.fc_drop(out_res))
        return pred, hidden, cell