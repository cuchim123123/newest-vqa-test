"""Configuration system using dataclasses + YAML loading."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class DataConfig:
    """Cấu hình liên quan đến dữ liệu."""
    hf_id: str = "HuggingFaceM4/A-OKVQA"
    train_ratio: float = 0.85
    freq_threshold: int = 3
    image_size: int = 224
    expand_rationales: bool = True

@dataclass
class ModelConfig:
    """Cấu hình kiến trúc mô hình."""
    embed_size: int = 300
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    use_pretrained_cnn: bool = True
    use_attention: bool = True

@dataclass
class TrainConfig:
    """Siêu tham số huấn luyện."""
    epochs: int = 20
    batch_size: int = 32 
    learning_rate: float = 3e-4
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    patience: int = 7
    beam_width: int = 10
    len_alpha: float = 0.6
    tf_start: float = 1.0
    tf_end: float = 0.0
    scheduler: str = "cosine"
    eta_min: float = 1e-6
    num_workers: int = 0
    pin_memory: bool = True
    eval_every: int = 2       
    warmup_epochs: int = 3    
    
    rep_penalty: float = 1.2
    min_gen_len: int = 5
    use_amp: bool = True

@dataclass
class Config:
    """Cấu hình tổng hợp cho dự án."""
    seed: int = 42
    device: str = "auto"
    
    log_dir: str = "/kaggle/working/logs"
    ckpt_dir: str = "/kaggle/working/checkpoints"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_variants: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Nạp cấu hình từ tệp YAML."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "seed" in raw: cfg.seed = raw["seed"]
        if "device" in raw: cfg.device = raw["device"]
        if "log_dir" in raw: cfg.log_dir = raw["log_dir"]
        if "ckpt_dir" in raw: cfg.ckpt_dir = raw["ckpt_dir"]

        sections = {"data": cfg.data, "model": cfg.model, "train": cfg.train}
        for section_name, section_obj in sections.items():
            if section_name in raw:
                for k, v in raw[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
                    else:
                        # Thêm cảnh báo nếu có key trong YAML nhưng thiếu trong dataclass
                        print(f"huộc tính '{k}' có trong YAML nhưng không tồn tại trong class {section_obj.__class__.__name__}")
        
        if "model_variants" in raw:
            cfg.model_variants = raw["model_variants"]

        return cfg

    def to_dict(self) -> dict:
        return asdict(self)