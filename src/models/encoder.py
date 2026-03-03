"""Image and Question encoders for VQA."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.data.dataset import PAD_IDX

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
            # DÙNG RESNET-50 THAY VÌ RESNET-18
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2]) 
            
            for p in list(self.cnn.parameters())[:-15]:
                p.requires_grad = False
                
            # Đảm bảo output luôn có kích thước không gian = 7x7 (giống bản scratch)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
                
            # LỚP NÉN: Giảm 2048 kênh của ResNet-50 xuống 512 kênh để tương thích với Decoder
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
            # Chặn cập nhật mean/var của BatchNorm trong ResNet để tránh phá hỏng weights đã train
            for m in self.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            # Re-normalize just in case inputs are [0, 1] standard tensors
            # This ensures stable inputs for the pretrained ResNet-50
            if images.max() <= 1.0:
                images = (images - self.mean) / self.std
        
        features = self.cnn(images)  
        if self.pretrained:
            features = self.adaptive_pool(features) # Đưa về dạng 7x7
            features = self.proj(features)
        B, C, H, W = features.size()
        return features.view(B, C, H * W).permute(0, 2, 1)

    def get_pretrained_params(self):
        """Returns parameters belonging to the pretrained backbone."""
        if self.pretrained:
            return self.cnn.parameters()
        return []

    def get_scratch_params(self):
        """Returns parameters that need to be learned from scratch."""
        if self.pretrained:
            # Everything except the CNN backbone
            return self.proj.parameters()
        return self.parameters()

# ... (Lớp QuestionEncoder giữ nguyên như file của bạn) ...
class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 300, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3, pretrained_emb: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, questions: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mask = (questions != PAD_IDX).float()
        emb = self.dropout(self.embedding(questions))
        cpu_lengths = lengths.detach().cpu().to(torch.int64).clamp(min=1)
        packed = pack_padded_sequence(emb, cpu_lengths, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=questions.size(1))
        return outputs, (h, c), mask