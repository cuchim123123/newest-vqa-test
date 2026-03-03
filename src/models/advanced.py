"""
Advanced adapters for VQA:
1. BertQuestionEncoder (Sử dụng Contextual Embeddings từ BERT thay thế GloVe)
2. BUTD_FasterRCNN_Encoder (Trích xuất đặc trưng vùng - Region Features từ Faster R-CNN)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class BertQuestionEncoder(nn.Module):
    """
    [Mục 3] Thay thế LSTM + GloVe bằng DistilBERT.
    Bert sẽ đọc toàn bộ câu văn để mã hóa theo ngữ cảnh (Contextual Embeddings).
    """
    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
            # Đóng băng khối BERT để tiết kiệm VRAM và tăng tốc huấn luyện (Khai thác pre-trained)
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # Chiếu từ không gian 768 chiều của DistilBERT xuống kích thước hidden_size của Decoder
            self.proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            )
        except ImportError:
            raise ImportError("Lỗi! Để dùng BERT, hãy chạy lệnh: pip install transformers")

    def forward(self, raw_questions: list[str], device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Tokenize trực tiếp khi mảng Text chạy qua forward
        encoded = self.tokenizer(raw_questions, padding=True, truncation=True, return_tensors='pt', max_length=50)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Đi qua mô hình BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Ép kiểu chiều Decoder chuẩn
        projected = self.proj(hidden_states) # (B, seq_len, hidden_size)
        
        # Lấy token đầu tiên (CLS token proxy ở DistilBert) làm vector tổng hợp ngữ nghĩa cho (h, c) của LSTM Decoder
        cls_feat = projected[:, 0, :] # (B, hidden_size)
        
        # Mô phỏng cấu trúc (h, c) 2 layers từ LSTM cũ để lắp ghép hoàn hảo với AnswerDecoder hiện tại
        # Kích thước cần thiết: (num_layers, B, hidden_size) -> (2, B, 512)
        h = cls_feat.unsqueeze(0).repeat(2, 1, 1).contiguous()
        c = torch.zeros_like(h).to(device)
        
        return projected, (h, c), attention_mask


class BUTD_FasterRCNN_Encoder(nn.Module):
    """
    [Mục 4] Bottom-Up Top-Down (BUTD) thay thế cho ResNet-50.
    Sử dụng Faster R-CNN để khoanh vùng 36 vật thể đáng chú ý nhất thay vì chia ảnh thành không gian lưới 7x7 ngẫu nhiên.
    """
    def __init__(self, out_dim: int = 512, max_regions: int = 36):
        super().__init__()
        import torchvision
        # Load mô hình Faster R-CNN đã được train trên COCO (Hỗ trợ phát hiện chó, mèo, cốc, xe...)
        self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.faster_rcnn.eval()
        
        # Đóng băng tham số
        for param in self.faster_rcnn.parameters():
            param.requires_grad = False
            
        self.max_regions = max_regions
        self.proj = nn.Linear(1024, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        LƯU Ý DÀNH CHO BẠN:
        Trích xuất RoI (Region of Interest) trực tiếp từ Torchvision cực kỳ tốn chi phí RAM lẫn thời gian train
        do số lượng bounding boxes trên mỗi ảnh là không giống nhau.
        Cách chuẩn SOTA: Bạn phải trích xuất (Extract Offline) trước thành các ma trận numpy (B, 36, 1024) 
        lưu cứng xuống ổ đĩa, rồi lúc train chỉ việc đọc lên! 
        """
        raise NotImplementedError("Để chạy full BUTD Pipeline mượt mà, nên tách riêng Feature Extractor ra chạy offline (thành file .npy/.h5) trên Kaggle trước khi đưa vào train LSTM.")
