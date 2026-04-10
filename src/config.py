"""Configuration system using dataclasses + YAML loading."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class DataConfig:
    """Configuration related to data."""
    hf_id: str = "HuggingFaceM4/A-OKVQA"
    train_ratio: float = 0.85
    freq_threshold: int = 3
    image_size: int = 224
    expand_rationales: bool = False

@dataclass
class ModelConfig:
    """Configuration related to model architecture."""
    embed_size: int = 300
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    use_pretrained_cnn: bool = True
    use_attention: bool = True
    bidirectional: bool = True
    num_answers: int = 0            # 0 = disable classification head
    cls_weight: float = 0.0         # disabled — generative only

@dataclass
class TrainConfig:
    """Configuration related to training."""
    epochs: int = 30
    batch_size: int = 48
    learning_rate: float = 3e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 6
    beam_width: int = 5
    len_alpha: float = 0.7
    tf_start: float = 1.0
    tf_end: float = 0.4
    scheduler: str = "cosine"
    eta_min: float = 1e-6
    num_workers: int = 0
    pin_memory: bool = True
    eval_every: int = 2
    warmup_epochs: int = 3

    rep_penalty: float = 1.5
    min_gen_len: int = 3
    use_amp: bool = True                      # FP16 — T4 Tensor Cores
    weight_decay: float = 5e-5
    pretrained_lr_ratio: float = 0.1
    unfreeze_after_epoch: int = 14
    prefetch_factor: int = 2

@dataclass
class Config:
    """Configuration for the entire project."""
    seed: int = 42
    device: str = "auto"                     # auto-detect CUDA/MPS/CPU

    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_variants: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from YAML file."""
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
                        # Add warning if key exists in YAML but is missing in dataclass
                        print(f"Property '{k}' exists in YAML but not in class {section_obj.__class__.__name__}")
        
        if "model_variants" in raw:
            cfg.model_variants = raw["model_variants"]

        return cfg

    def to_dict(self) -> dict:
        return asdict(self)