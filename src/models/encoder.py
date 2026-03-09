"""Image and Question encoders for VQA."""

from __future__ import annotations
from typing import Optional
import logging
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.data.dataset import PAD_IDX

logger = logging.getLogger("VQA")

class CNNEncoder(nn.Module):
    """Image feature extractor. Outputs a spatial grid instead of a flat vector."""
    CNN_OUT_DIM: int = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        
        # ImageNet standardization stats
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if pretrained:
            # USE RESNET-50 INSTEAD OF RESNET-18
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2]) 
            
            # Freeze ALL ResNet backbone by default — only proj layer trains
            for p in self.cnn.parameters():
                p.requires_grad = False
                
            # Ensure spatial output size = 7x7 (like scratch version)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
                
            # COMPRESSION LAYER: Reduce ResNet-50 channels from 2048 to 512 to match Decoder
            self.proj = nn.Conv2d(2048, self.CNN_OUT_DIM, kernel_size=1)
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, self.CNN_OUT_DIM, 3, padding=1), nn.BatchNorm2d(self.CNN_OUT_DIM), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((7, 7)),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained:
            # Block ResNet BatchNorm mean/var updates to avoid damaging trained weights
            for m in self.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # NOTE: Images are already normalized by the DataLoader transforms
        # (Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
        # so we do NOT re-normalize here.
        features = self.cnn(images)  
        if self.pretrained:
            features = self.adaptive_pool(features) # Bring to 7x7
            features = self.proj(features)
        B, C, H, W = features.size()
        return features.view(B, C, H * W).permute(0, 2, 1)

    def unfreeze_backbone(self, num_layers: int = 1):
        """Unfreeze the last `num_layers` ResNet stages (layer1..layer4).
        
        ResNet-50 children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
        Indices 4-7 are the residual stages.  num_layers=1 unfreezes layer4 only.
        BatchNorm layers are kept frozen to preserve pretrained statistics.
        """
        if not self.pretrained:
            return
        children = list(self.cnn.children())
        # last num_layers stages (from the end)
        stages_to_unfreeze = children[max(4, 8 - num_layers):]
        count = 0
        for stage in stages_to_unfreeze:
            for name, p in stage.named_parameters():
                # Skip BatchNorm params — keep them frozen
                if 'bn' in name or 'downsample.1' in name:  # downsample.1 is BN in shortcut
                    continue
                p.requires_grad = True
                count += 1
        logger.info(f"  Unfroze {count} params in last {num_layers} ResNet stage(s) (BN kept frozen)")

    def get_pretrained_params(self):
        """Returns trainable parameters of the pretrained backbone."""
        if self.pretrained:
            return [p for p in self.cnn.parameters() if p.requires_grad]
        return []

    def get_scratch_params(self):
        """Returns parameters that need to be learned from scratch."""
        if self.pretrained:
            # proj layer + any non-cnn params
            return list(self.proj.parameters())
        return list(self.parameters())

# ... (QuestionEncoder class remains the same as your file) ...
class QuestionEncoder(nn.Module):
    """Encodes questions with a multi-layer **Bidirectional** LSTM.

    tokens → Embedding(GloVe) → Dropout → BiLSTM → LayerNorm → projection

    The bidirectional LSTM captures both forward and backward context for each
    token.  The outputs are projected from ``hidden_size*2`` back to
    ``hidden_size`` so that the decoder can consume them without dimension
    changes.  The final hidden state is similarly projected.
    """

    def __init__(
        self, vocab_size: int, embed_size: int = 300,
        hidden_size: int = 512, num_layers: int = 2,
        dropout: float = 0.3, bidirectional: bool = True,
        pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True  # fine-tune

        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Normalize bi-directional outputs before projection
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)

        # Project BiLSTM outputs (H*2) → H so downstream modules stay unchanged
        self.output_proj: nn.Module = (
            nn.Linear(hidden_size * 2, hidden_size)
            if bidirectional else nn.Identity()
        )
        # Project concatenated fwd+bwd final hidden state → H per layer
        self.hidden_proj: nn.Module = (
            nn.Linear(hidden_size * 2, hidden_size)
            if bidirectional else nn.Identity()
        )

    def forward(
        self, questions: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            questions: ``(B, seq_len)`` token indices.
            lengths:   ``(B,)`` actual lengths (before padding).

        Returns:
            outputs: ``(B, seq_len, hidden_size)`` — projected BiLSTM output.
            (h, c):  decoder-ready hidden state, each ``(num_layers, B, hidden_size)``.
            mask:    ``(B, seq_len)`` float mask (1.0 for real tokens).
        """
        mask = (questions != PAD_IDX).float()
        emb = self.dropout(self.embedding(questions))

        cpu_lengths = lengths.detach().cpu().to(torch.int64).clamp(min=1)
        packed = pack_padded_sequence(emb, cpu_lengths, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=questions.size(1))

        # LayerNorm on raw BiLSTM outputs, then project to hidden_size
        outputs = self.output_proj(self.layer_norm(outputs))  # (B, T, H)

        if self.bidirectional:
            # h shape: (num_layers*2, B, H) → merge directions for each layer
            # Reshape to (num_layers, 2, B, H), concat directions, project
            h = h.view(self.num_layers, 2, -1, self.hidden_size)
            c = c.view(self.num_layers, 2, -1, self.hidden_size)
            # Concatenate fwd and bwd → (num_layers, B, H*2) → project → (num_layers, B, H)
            h = self.hidden_proj(torch.cat([h[:, 0], h[:, 1]], dim=-1))
            c = self.hidden_proj(torch.cat([c[:, 0], c[:, 1]], dim=-1))

        return outputs, (h, c), mask