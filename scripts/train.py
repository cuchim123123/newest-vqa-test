"""
CLI script: Train VQA model variants.
Hỗ trợ huấn luyện đồng thời nhiều biến thể (M1, M2, M3, M4) với cơ chế Spatial Attention.
"""

from __future__ import annotations

import argparse
import os
import sys
import gc
import random

# Thêm gốc dự án vào sys.path để nhận diện thư mục src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset

from src.config import Config
from src.data.preprocessing import extract_answer, expand_data_with_rationales
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.data.glove import download_glove, load_glove_embeddings
from src.models.vqa_model import VQAModel
from src.engine.trainer import train_model
from src.utils.helpers import get_device, set_seed, setup_logging


def main() -> None:
    # ── 1. CẤU HÌNH ARGUMENTS ──
    parser = argparse.ArgumentParser(description="VQA Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--models", nargs="+", help="Specific model names to train (e.g., M4_Pretrained_Attn)")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    args = parser.parse_args()

    # ── 2. KHỞI TẠO MÔI TRƯỜNG ──
    cfg = Config.from_yaml(args.config)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        
    set_seed(cfg.seed)
    logger = setup_logging(cfg.log_dir)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)

    logger.info(f"Starting training on device: {device}")
    logger.info(f"Batch size: {cfg.train.batch_size} | Freq Threshold: {cfg.data.freq_threshold}")

    # ── 3. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ──
    logger.info("Loading A-OKVQA dataset from HuggingFace...")
    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Xây dựng Vocab (ưu tiên dùng rationales để sinh câu trả lời dài)
    all_questions = [item["question"] for item in hf_train] + [item["question"] for item in hf_val]
    all_answers = []
    for item in (list(hf_train) + list(hf_val)):
        rationales = item.get("rationales", [])
        if rationales:
            all_answers.extend(rationales)
        else:
            all_answers.append(extract_answer(item))

    question_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    question_vocab.build_vocabulary(all_questions)
    answer_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    answer_vocab.build_vocabulary(all_answers)
    
    logger.info(f"Vocab sizes: Question={len(question_vocab)}, Answer={len(answer_vocab)}")

    # Chia tách dữ liệu (85% train / 15% val nội bộ)
    hf_train_list = list(hf_train)
    random.shuffle(hf_train_list)
    split_idx = int(len(hf_train_list) * cfg.data.train_ratio)
    
    train_data_raw = hf_train_list[:split_idx]
    val_data = hf_train_list[split_idx:]

    # Nhân bản dữ liệu với rationales (Cách 2)
    if cfg.data.expand_rationales:
        train_data = expand_data_with_rationales(train_data_raw)
        logger.info(f"Expanded training samples: {len(train_data_raw)} -> {len(train_data)}")
    else:
        train_data = train_data_raw

    # ── 4. CHUẨN BỊ DATALOADER ──
    train_transform = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = AOKVQA_Dataset(train_data, question_vocab, answer_vocab, train_transform)
    val_ds = AOKVQA_Dataset(val_data, question_vocab, answer_vocab, val_transform)

    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "collate_fn": collate_fn,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── 5. TẢI GLOVE EMBEDDINGS ──
    download_glove()
    q_glove = load_glove_embeddings(question_vocab, cfg.model.embed_size)
    a_glove = load_glove_embeddings(answer_vocab, cfg.model.embed_size)

    # ── 6. VÒNG LẶP HUẤN LUYỆN CÁC BIẾN THỂ ──
    variants_to_train = args.models or list(cfg.model_variants.keys())
    
    for name in variants_to_train:
        logger.info(f"\n{'='*50}\nTraining Variant: {name}\n{'='*50}")
        
        # Giải phóng bộ nhớ trước khi nạp model mới (Đặc biệt quan trọng cho Mac/MPS)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Khởi tạo model dựa trên config biến thể
        variant_cfg = cfg.model_variants[name]
        model = VQAModel(
            q_vocab_size=len(question_vocab),
            a_vocab_size=len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            q_pretrained_emb=q_glove,
            a_pretrained_emb=a_glove,
            **variant_cfg
        ).to(device)

        # Lưu lại vocab để dùng cho inference sau này
        os.makedirs(os.path.dirname("data/processed/vocab.pth"), exist_ok=True)
        torch.save({"q_vocab": question_vocab, "a_vocab": answer_vocab}, "data/processed/vocab.pth")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
        if os.path.exists(ckpt_path) and not getattr(args, 'force', False):
            logger.info(f"⏭️  Bỏ qua {name}: checkpoint đã tồn tại tại {ckpt_path}")
            model.cpu()
            continue

        # Gọi hàm train từ engine
        train_model(
            model=model,
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            answer_vocab=answer_vocab,
            device=device,
            epochs=cfg.train.epochs,
            lr=cfg.train.learning_rate,
            ckpt_dir=cfg.ckpt_dir,
            patience=cfg.train.patience,
            tf_end=cfg.train.tf_end,
            eval_every=cfg.train.eval_every,
            use_amp=cfg.train.use_amp  # MPS không hỗ trợ tốt AMP
        )
        
        # Di chuyển model về CPU sau khi train xong để tiết kiệm VRAM cho biến thể tiếp theo
        model.cpu()

    logger.info("All training tasks completed successfully.")

if __name__ == "__main__":
    main()