"""
Advanced adapters for VQA:
1. BertQuestionEncoder (Using Contextual Embeddings from BERT instead of GloVe)
2. BUTD_FasterRCNN_Encoder (Extracting region features from Faster R-CNN)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class BertQuestionEncoder(nn.Module):
    """
    [Section 3] Replace LSTM + GloVe with DistilBERT.
    Bert will read the entire sentence to encode contextually (Contextual Embeddings).
    """
    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
            # Freeze BERT block to save VRAM and speed up training (Exploit pre-trained)
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # Project 768-dim DistilBERT space to Decoder's hidden_size
            self.proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            )
        except ImportError:
            raise ImportError("Error! To use BERT, please run: pip install transformers")

    def forward(self, raw_questions: list[str], device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Tokenize directly when Text array passes through forward
        encoded = self.tokenizer(raw_questions, padding=True, truncation=True, return_tensors='pt', max_length=50)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Pass through BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Cast to standard Decoder shape
        projected = self.proj(hidden_states) # (B, seq_len, hidden_size)
        
        # Take first token (CLS proxy in DistilBert) as semantic vector for LSTM Decoder's (h, c)
        cls_feat = projected[:, 0, :] # (B, hidden_size)
        
        # Simulate (h, c) 2-layer structure from old LSTM to perfectly match current AnswerDecoder
        # Required shape: (num_layers, B, hidden_size) -> (2, B, 512)
        h = cls_feat.unsqueeze(0).repeat(2, 1, 1).contiguous()
        c = torch.zeros_like(h).to(device)
        
        return projected, (h, c), attention_mask


class BUTD_FasterRCNN_Encoder(nn.Module):
    """
    [Section 4] Bottom-Up Top-Down (BUTD) replaces ResNet-50.
    Using Faster R-CNN to extract 36 most notable objects from image instead of randomly splitting into 7x7 grid.
    """
    def __init__(self, out_dim: int = 512, max_regions: int = 36):
        super().__init__()
        import torchvision
        # Load Faster R-CNN trained on COCO (Supports dog, cat, cup, car detection...)
        self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.faster_rcnn.eval()
        
        # Freeze parameters
        for param in self.faster_rcnn.parameters():
            param.requires_grad = False
            
        self.max_regions = max_regions
        self.proj = nn.Linear(1024, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Extracting RoI (Region of Interest) directly from Torchvision is extremely expensive in terms of VRAM and training time
        due to the number of bounding boxes on each image being different.
        Standard SOTA approach: You must extract (Extract Offline) first into numpy arrays (B, 36, 1024) save to disk, then during training just read them up! 
        """
        raise NotImplementedError("To run full BUTD Pipeline smoothly, you should extract (Extract Offline) first into numpy arrays (B, 36, 1024) save to disk, then during training just read them up!")
