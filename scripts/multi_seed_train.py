"""Multi-seed training for statistical significance."""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.preprocessing import extract_answer, expand_data_with_rationales
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.data.glove import download_glove, load_glove_embeddings
from src.models.vqa_model import VQAModel
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model
from src.utils.helpers import get_device, set_seed, setup_logging

def run_single_seed(cfg: Config, seed: int, device: torch.device, variants: list[str]) -> dict[str, dict[str, float]]:
    """Run the full train + eval pipeline for a specific seed."""
    set_seed(seed)
    logger = setup_logging(cfg.log_dir)
    logger.info(f"\n{'#' * 40}\n#  RUNNING SEED: {seed}\n{'#' * 40}")

    # Transforms & Data Loading
    transform_eval = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Build Vocab (Fixed threshold=3)
    all_text = [item["question"] for item in hf_train] + [item["question"] for item in hf_val]
    q_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    q_vocab.build_vocabulary(all_text)
    
    a_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    a_vocab.build_vocabulary([extract_answer(item) for item in hf_train])

    # Shuffle and Split based on current seed
    train_list = list(hf_train)
    random.shuffle(train_list)
    split_idx = int(len(train_list) * cfg.data.train_ratio)
    
    train_ds = AOKVQA_Dataset(expand_data_with_rationales(train_list[:split_idx]), q_vocab, a_vocab, transform_eval)
    val_ds = AOKVQA_Dataset(train_list[split_idx:], q_vocab, a_vocab, transform_eval)
    test_ds = AOKVQA_Dataset(list(hf_val), q_vocab, a_vocab, transform_eval)

    loader_args = {'batch_size': cfg.train.batch_size, 'collate_fn': collate_fn, 'num_workers': 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    download_glove()
    q_emb = load_glove_embeddings(q_vocab, cfg.model.embed_size)
    a_emb = load_glove_embeddings(a_vocab, cfg.model.embed_size)

    seed_results = {}
    for name in variants:
        gc.collect()
        if device.type == "mps": torch.mps.empty_cache()

        variant_model_cfg = {k: v for k, v in cfg.model_variants[name].items() if k != "train_overrides"}
        model = VQAModel(
            q_vocab_size=len(q_vocab),
            a_vocab_size=len(a_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional,
            num_answers=cfg.model.num_answers,
            q_pretrained_emb=q_emb,
            a_pretrained_emb=a_emb,
            **variant_model_cfg,
        ).to(device)
        
        import copy
        variant_train_cfg = copy.deepcopy(cfg.train.__dict__)
        variant_train_cfg.update(cfg.model_variants[name].get("train_overrides", {}))
        train_model(
            model=model, name=f"{name}_s{seed}",
            train_loader=train_loader, val_loader=val_loader,
            answer_vocab=a_vocab, device=device,
            epochs=variant_train_cfg["epochs"],
            lr=variant_train_cfg["learning_rate"],
            ckpt_dir=cfg.ckpt_dir,
            label_smoothing=variant_train_cfg["label_smoothing"],
            patience=variant_train_cfg["patience"],
            grad_clip=variant_train_cfg["grad_clip"],
            tf_start=variant_train_cfg["tf_start"],
            tf_end=variant_train_cfg["tf_end"],
            warmup_epochs=variant_train_cfg["warmup_epochs"],
            eval_every=variant_train_cfg["eval_every"],
            use_amp=variant_train_cfg["use_amp"],
            cls_weight=cfg.model.cls_weight,
            weight_decay=variant_train_cfg["weight_decay"],
            pretrained_lr_ratio=variant_train_cfg["pretrained_lr_ratio"],
            unfreeze_after_epoch=variant_train_cfg["unfreeze_after_epoch"],
        )
        
        eval_res = evaluate_model(model, test_loader, a_vocab, q_vocab, device, cfg.ckpt_dir, f"{name}_s{seed}")
        seed_results[name] = eval_res["metrics"]
        
    return seed_results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()
    variants = list(cfg.model_variants.keys())
    seeds = [42 + i * 100 for i in range(args.n_seeds)]

    all_results = [run_single_seed(cfg, s, device, variants) for s in seeds]

    # Compute statistics
    print("\n" + "="*80 + "\nSTATISTICAL RESULTS (Mean ± Std)\n" + "="*80)
    for name in variants:
        f1_vals = [res[name]['f1'] for res in all_results]
        acc_vals = [res[name]['accuracy'] for res in all_results]
        print(f"{name:<25}: F1={np.mean(f1_vals):.4f}±{np.std(f1_vals):.4f} | Acc={np.mean(acc_vals):.4f}±{np.std(acc_vals):.4f}")

if __name__ == "__main__":
    main()