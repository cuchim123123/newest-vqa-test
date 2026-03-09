from __future__ import annotations

import argparse
import os
import sys
import gc
import random

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
    # ── 1. ARGUMENTS CONFIGURATION ──
    parser = argparse.ArgumentParser(description="VQA Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--models", nargs="+", help="Specific model names to train (e.g., M4_Pretrained_Attn)")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    args = parser.parse_args()

    # ── 2. ENVIRONMENT INITIALIZATION ──
    cfg = Config.from_yaml(args.config)
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        
    set_seed(cfg.seed)
    logger = setup_logging(cfg.log_dir)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)

    logger.info(f"Starting training on device: {device}")
    logger.info(f"Batch size: {cfg.train.batch_size} | Freq Threshold: {cfg.data.freq_threshold}")

    # ── 3. DATA LOADING AND PREPROCESSING ──
    logger.info("Loading A-OKVQA dataset from HuggingFace...")
    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Build Vocab (prioritize rationales for generating long answers)
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

    # Split Data (85% train / 15% internal val)
    hf_train_list = list(hf_train)
    random.shuffle(hf_train_list)
    split_idx = int(len(hf_train_list) * cfg.data.train_ratio)
    
    train_data_raw = hf_train_list[:split_idx]
    val_data = hf_train_list[split_idx:]

    # Duplicate data with rationales (Method 2)
    if cfg.data.expand_rationales:
        train_data = expand_data_with_rationales(train_data_raw)
        logger.info(f"Expanded training samples: {len(train_data_raw)} -> {len(train_data)}")
    else:
        train_data = train_data_raw

    # ── 4. DATALOADER PREPARATION ──
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

    # ── 5. LOAD GLOVE EMBEDDINGS ──
    download_glove()
    q_glove = load_glove_embeddings(question_vocab, cfg.model.embed_size)
    a_glove = load_glove_embeddings(answer_vocab, cfg.model.embed_size)

    # ── 6. VARIANTS TRAINING LOOP ──
    variants_to_train = args.models or list(cfg.model_variants.keys())
    
    for name in variants_to_train:
        logger.info(f"\n{'='*50}\nTraining Variant: {name}\n{'='*50}")
        
        # Free memory before loading new model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Initialize model based on variant config
        variant_cfg = cfg.model_variants[name]
        model = VQAModel(
            q_vocab_size=len(question_vocab),
            a_vocab_size=len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
            num_answers=cfg.model.num_answers,
            q_pretrained_emb=q_glove,
            a_pretrained_emb=a_glove,
            **variant_cfg
        ).to(device)

        # Save vocab for future inference
        os.makedirs(os.path.dirname("data/processed/vocab.pth"), exist_ok=True)
        torch.save({"q_vocab": question_vocab, "a_vocab": answer_vocab}, "data/processed/vocab.pth")

        ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
        if os.path.exists(ckpt_path) and not getattr(args, 'force', False):
            logger.info(f"Skip training {name}: checkpoint already exists at {ckpt_path}")
            model.cpu()
            continue

        # Call train function from engine
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
            label_smoothing=cfg.train.label_smoothing,
            patience=cfg.train.patience,
            grad_clip=cfg.train.grad_clip,
            tf_start=cfg.train.tf_start,
            tf_end=cfg.train.tf_end,
            warmup_epochs=cfg.train.warmup_epochs,
            eval_every=cfg.train.eval_every,
            use_amp=cfg.train.use_amp,
            cls_weight=cfg.model.cls_weight,
            answer_to_idx=getattr(answer_vocab, 'stoi', None),
        )
        
        # Move model to CPU after training to save VRAM for next variant
        model.cpu()

    logger.info("All training tasks completed successfully.")

if __name__ == "__main__":
    main()