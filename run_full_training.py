"""
Full Training Pipeline — runs all 4 VQA model variants end-to-end.
Usage: python run_full_training.py
"""

import sys
import os
import random
import gc
import json
import time
import logging
import copy

# ── Setup path ──────────────────────────────────────────────────
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# ── Core imports ────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt

# NLP
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Project modules
from src.config import Config
from src.data.glove import download_glove, load_glove_embeddings
from src.data.preprocessing import extract_answer
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.models.vqa_model import VQAModel
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model
from src.utils.helpers import get_device, set_seed, setup_logging, decode_sequence
from src.utils.visualization import (
    plot_training_curves,
    plot_radar_chart,
    plot_bar_chart,
    plot_question_type_analysis,
    plot_confusion_matrix,
    visualize_attention,
    visualize_attention_overlay,
)
from src.engine.evaluator import evaluate_by_question_type

# ── Configuration ───────────────────────────────────────────────
cfg = Config.from_yaml(os.path.join(project_path, "configs/default.yaml"))
set_seed(cfg.seed)
device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
logger = setup_logging(cfg.log_dir)

# ── CUDA Performance Tuning (T4 Tensor Core optimizations) ─────
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True           # auto-tune convolution kernels
    # TF32 settings (compatible with PyTorch 2.9+)
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    except (AttributeError, TypeError):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
else:
    torch.backends.cudnn.benchmark = True
    _gpu_name = "N/A"
    _gpu_mem  = 0

print(f"\n{'='*60}")
print(f"  VQA FULL TRAINING PIPELINE (Kaggle T4 Optimized)")
print(f"{'='*60}")
print(f"  GPU:          {_gpu_name} ({_gpu_mem:.1f} GB)")
print(f"  Device:       {device}")
print(f"  Batch Size:   {cfg.train.batch_size}")
print(f"  Epochs:       {cfg.train.epochs}  |  Eval every {cfg.train.eval_every}")
print(f"  AMP (FP16):   {cfg.train.use_amp}")
print(f"  Workers:      {cfg.train.num_workers}  |  Prefetch: {cfg.train.prefetch_factor}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA & BUILD VOCABULARY
# ═══════════════════════════════════════════════════════════════
IMG_SIZE = cfg.data.image_size

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MAX_TRAIN = 15000
MAX_VAL   = 2500
MAX_TEST  = 1070

# ── Load dataset (offline-first for Kaggle speed) ─────────────
# If a pre-saved copy exists in /kaggle/input, load from disk instantly.
# Otherwise download from HuggingFace Hub (slow on Kaggle).
_LOCAL_DS_DIRS = [
    os.path.join(project_path, "data", "aokvqa_hf"),       # local dev
    "/kaggle/input/aokvqa-dataset/aokvqa_hf",               # Kaggle input dataset
    "/kaggle/input/aokvqa-dataset",                          # alt layout
]

_local_ds_path = None
for _d in _LOCAL_DS_DIRS:
    if os.path.isdir(_d):
        _local_ds_path = _d
        break

if _local_ds_path:
    print(f"Loading dataset OFFLINE from {_local_ds_path} ...", flush=True)
    from datasets import load_from_disk
    _ds = load_from_disk(_local_ds_path)
    hf_train = _ds["train"]
    hf_val   = _ds["validation"]
else:
    print(f"Loading dataset ONLINE: {cfg.data.hf_id} ...", flush=True)
    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val   = load_dataset(cfg.data.hf_id, split="validation")

    # Auto-save for future offline use
    _save_path = os.path.join(project_path, "data", "aokvqa_hf")
    try:
        from datasets import DatasetDict
        DatasetDict({"train": hf_train, "validation": hf_val}).save_to_disk(_save_path)
        print(f"Saved dataset to {_save_path} for future offline use", flush=True)
    except Exception as e:
        print(f"Warning: Could not cache dataset locally: {e}", flush=True)

print(f"Full HF dataset: train={len(hf_train)}, val={len(hf_val)}", flush=True)

# Subsample indices
n_total = len(hf_train)
indices = list(range(n_total))
random.shuffle(indices)

split_idx = int(min(n_total, MAX_TRAIN + MAX_VAL) * cfg.data.train_ratio)
train_indices = indices[:min(split_idx, MAX_TRAIN)]
val_indices   = indices[split_idx:split_idx + MAX_VAL]
test_indices  = list(range(min(len(hf_val), MAX_TEST)))

print(f"Subsampled: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

# Build Vocabulary (text only)
text_cols = [c for c in hf_train.column_names if c != "image"]
all_questions, all_answers = [], []

for idx in tqdm(train_indices + val_indices, desc="Vocab (train+val)", leave=False, mininterval=2.0):
    row = hf_train.select_columns(text_cols)[idx]
    all_questions.append(row["question"])
    all_answers.append(extract_answer(row))

# NOTE: Test set is intentionally EXCLUDED from vocabulary building
# to prevent data leakage. Test-only words will map to <UNK>.

question_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
question_vocab.build_vocabulary(all_questions)
answer_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
answer_vocab.build_vocabulary(all_answers)
del all_questions, all_answers
gc.collect()

print(f"Question Vocab: {len(question_vocab):,} | Answer Vocab: {len(answer_vocab):,}")

# Build records (text only, with HF index)
train_text_ds = hf_train.select_columns(text_cols)
val_text_ds   = hf_val.select_columns(text_cols)

train_records = [dict(train_text_ds[i], _hf_idx=i) for i in tqdm(train_indices, desc="train recs", leave=False, mininterval=2.0)]
val_records   = [dict(train_text_ds[i], _hf_idx=i) for i in tqdm(val_indices, desc="val recs", leave=False, mininterval=2.0)]
test_records  = [dict(val_text_ds[i], _hf_idx=i)   for i in tqdm(test_indices, desc="test recs", leave=False, mininterval=2.0)]

del train_text_ds, val_text_ds
gc.collect()

# Lazy-image Dataset wrapper
class LazyImageDataset(AOKVQA_Dataset):
    """Loads images on-the-fly from HF Arrow dataset."""
    def __init__(self, records, hf_ds, q_vocab, a_vocab, transform):
        super().__init__(records, q_vocab, a_vocab, transform)
        self.hf_ds = hf_ds

    def __getitem__(self, idx):
        item = self.data[idx]
        hf_idx = item.get("_hf_idx", idx)
        pil_image = self.hf_ds[hf_idx]["image"]
        item["image"] = pil_image
        result = super().__getitem__(idx)
        item["image"] = None
        return result

train_dataset = LazyImageDataset(train_records, hf_train, question_vocab, answer_vocab, transform_train)
val_dataset   = LazyImageDataset(val_records,   hf_train, question_vocab, answer_vocab, transform_eval)
test_dataset  = LazyImageDataset(test_records,  hf_val,   question_vocab, answer_vocab, transform_eval)

# GloVe Embeddings
download_glove()
q_glove_emb = load_glove_embeddings(question_vocab, embed_dim=cfg.model.embed_size)
a_glove_emb = load_glove_embeddings(answer_vocab,   embed_dim=cfg.model.embed_size)

print(f"\n✓ Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
print(f"✓ GloVe loaded.")

# ═══════════════════════════════════════════════════════════════
# STEP 2: CREATE DATALOADERS
# ═══════════════════════════════════════════════════════════════
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

BATCH_SIZE = cfg.train.batch_size
_num_workers = 0 if IS_KAGGLE else cfg.train.num_workers
_pin_memory = (not IS_KAGGLE) and cfg.train.pin_memory and (device.type == 'cuda')
_prefetch = cfg.train.prefetch_factor if _num_workers > 0 else None
_persistent = _num_workers > 0  # keep workers alive between epochs

if IS_KAGGLE:
    print("ℹ️  Kaggle detected: num_workers=0, pin_memory=False (hang prevention)")

loader_common = {
    "batch_size": BATCH_SIZE,
    "collate_fn": collate_fn,
    "num_workers": _num_workers,
    "pin_memory": _pin_memory,
    "prefetch_factor": _prefetch,
    "persistent_workers": _persistent,
}

train_loader = DataLoader(train_dataset, shuffle=True,  drop_last=True, **loader_common)
val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_common)
test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_common)

print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

Q_VOCAB = len(question_vocab)
A_VOCAB = len(answer_vocab)
print(f"Q vocab: {Q_VOCAB:,} | A vocab: {A_VOCAB:,}")
print(f"Variants to train: {list(cfg.model_variants.keys())}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: TRAIN ALL 4 VARIANTS
# ═══════════════════════════════════════════════════════════════
print(f"\n⚡ Training {len(train_dataset):,} samples, {cfg.train.epochs} epochs, "
      f"patience={cfg.train.patience}, eval_every={cfg.train.eval_every}\n")

all_histories = {}
train_start = time.time()

FORCE_RETRAIN = True  # Set False to skip already-trained variants

for name, variant_cfg in cfg.model_variants.items():
    ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
    resume_path = os.path.join(cfg.ckpt_dir, f"resume_{name}.pth")

    # Smart skip logic:
    #  - If a resume checkpoint exists → ALWAYS enter training (trainer.py resumes)
    #  - If only best exists + FORCE_RETRAIN=False → skip (already trained)
    #  - If only best exists + FORCE_RETRAIN=True → retrain from scratch
    if os.path.exists(resume_path):
        print(f"⏩ {name} has resume checkpoint — will continue training.")
    elif os.path.exists(ckpt_path) and not FORCE_RETRAIN:
        print(f"✓ {name} checkpoint exists, skipping (set FORCE_RETRAIN=True to override).")
        continue
    elif os.path.exists(ckpt_path) and FORCE_RETRAIN:
        print(f"⚠️  {name} checkpoint exists but FORCE_RETRAIN=True — retraining.")

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {name}")
    print(f"{'='*60}")

    variant_model_cfg = {k: v for k, v in variant_cfg.items() if k != "train_overrides"}
    variant_train_cfg = copy.deepcopy(cfg.train.__dict__)
    variant_train_cfg.update(variant_cfg.get("train_overrides", {}))
    if variant_cfg.get("train_overrides"):
        print(f"  train_overrides: {variant_cfg['train_overrides']}")

    model = VQAModel(
        q_vocab_size=Q_VOCAB,
        a_vocab_size=A_VOCAB,
        embed_size=cfg.model.embed_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
        num_answers=cfg.model.num_answers,
        q_pretrained_emb=q_glove_emb,
        a_pretrained_emb=a_glove_emb,
        **variant_model_cfg,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.to(device)

    t0 = time.time()
    hist = train_model(
        model=model,
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_vocab=answer_vocab,
        device=device,
        epochs=variant_train_cfg["epochs"],
        lr=variant_train_cfg["learning_rate"],
        use_beam=False,
        beam_w=variant_train_cfg["beam_width"],
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
    elapsed = time.time() - t0
    all_histories[name] = hist
    print(f"✓ {name} done in {elapsed/60:.1f} min. GPU freed.\n")

    model.cpu()
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

total_train_time = time.time() - train_start
print(f"\n✅ All training complete in {total_train_time/3600:.1f} hours.")

# Save training histories
hist_path = os.path.join(project_path, "results")
os.makedirs(hist_path, exist_ok=True)
for name, hist in all_histories.items():
    with open(os.path.join(hist_path, f"{name}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
print(f"✓ Training histories saved to {hist_path}/")

# ═══════════════════════════════════════════════════════════════
# STEP 4: REBUILD MODELS FROM CHECKPOINTS & EVALUATE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📊 EVALUATION PHASE")
print(f"{'='*60}\n")

models_dict = {}
for name, variant_cfg in cfg.model_variants.items():
    ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
    if os.path.exists(ckpt_path):
        variant_model_cfg = {k: v for k, v in variant_cfg.items() if k != "train_overrides"}
        m = VQAModel(
            q_vocab_size=Q_VOCAB, a_vocab_size=A_VOCAB,
            embed_size=cfg.model.embed_size, hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers, dropout=cfg.model.dropout,
            bidirectional=cfg.model.bidirectional, num_answers=cfg.model.num_answers,
            **variant_model_cfg,
        )
        models_dict[name] = m
        print(f"✓ {name} ready (checkpoint found)")
    else:
        print(f"⚠️  {name} — no checkpoint")

print(f"\n{len(models_dict)} models ready for evaluation.\n")

test_results = {}
all_eval_data = {}

for name in models_dict:
    ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
    if not os.path.exists(ckpt_path):
        continue

    model = models_dict[name]
    model.to(device)

    print(f"✓ Evaluating: {name} (beam={cfg.train.beam_width}, "
          f"len_alpha={cfg.train.len_alpha})")

    try:
        eval_data = evaluate_model(
            model=model,
            test_loader=test_loader,
            answer_vocab=answer_vocab,
            question_vocab=question_vocab,
            device=device,
            ckpt_dir=cfg.ckpt_dir,
            name=name,
            beam_width=cfg.train.beam_width,
            len_alpha=cfg.train.len_alpha,
            rep_penalty=cfg.train.rep_penalty,
            min_gen_len=cfg.train.min_gen_len,
        )
        test_results[name] = eval_data["metrics"]
        all_eval_data[name] = eval_data
        print(f"  ✓ F1={eval_data['metrics'].get('f1', 0):.4f}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

    model.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Print comparison table
if test_results:
    print(f"\n{'═'*125}")
    print(f"║ {'Model':<25s} │ {'Semantic':>10s} │ {'Acc':>8s} │ "
          f"{'F1':>8s} │ {'METEOR':>8s} │ {'BLEU-4':>8s} ║")
    print(f"╟{'─'*27}┼{'─'*12}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}╢")

    for name, m in test_results.items():
        sem = m.get("semantic", 0.0)
        print(f"║ {name:<25s} │ {sem:>10.4f} │ {m['accuracy']:>8.4f} │ "
              f"{m['f1']:>8.4f} │ {m['meteor']:>8.4f} │ {m['bleu4']:>8.4f} ║")

    print(f"{'═'*125}")

    best_name = max(test_results, key=lambda k: test_results[k]["f1"])
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   F1={test_results[best_name]['f1']:.4f}, "
          f"Semantic={test_results[best_name].get('semantic', 0.0):.4f}")

    # Save test results
    with open(os.path.join(hist_path, "test_results.json"), "w") as f:
        json.dump({k: v for k, v in test_results.items()}, f, indent=2)
    print(f"✓ Test results saved to {hist_path}/test_results.json")
else:
    print("⚠️  No test results.")

# ═══════════════════════════════════════════════════════════════
# STEP 5: GENERATE VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"📈 GENERATING VISUALIZATIONS")
print(f"{'='*60}\n")

try:
    if all_histories:
        print("✓ Training curves...")
        plot_training_curves(all_histories, save_prefix="fig_training_")

    if test_results:
        print("✓ Radar chart...")
        plot_radar_chart(test_results, save_path="fig_radar_comparison.png")
        print("✓ Bar chart...")
        plot_bar_chart(test_results, save_path="fig_f1_bar_chart.png")

        if best_name in all_eval_data:
            best_data = all_eval_data[best_name]
            print(f"✓ Question type analysis for {best_name}...")
            qtype_results = evaluate_by_question_type(
                best_data["preds"], best_data["refs"], best_data["questions"]
            )
            plot_question_type_analysis(qtype_results, save_path="fig_qtype_analysis.png")
            plot_confusion_matrix(
                best_data["preds"], best_data["refs"], best_data["questions"],
                save_path="fig_confusion_matrix.png",
            )

        # Attention visualization
        attn_name = None
        for candidate in ["M4_Pretrained_Attn", "M2_Scratch_Attn"]:
            if candidate in models_dict:
                attn_name = candidate
                break

        if attn_name:
            model_viz = models_dict[attn_name]
            if hasattr(model_viz, "use_attention") and model_viz.use_attention:
                model_viz.to(device)
                print(f"✓ Attention visualization for {attn_name}...")
                try:
                    visualize_attention(
                        model_viz, test_loader, answer_vocab, question_vocab,
                        device, n=3, save_path="fig_attention_heatmap.png",
                    )
                except Exception as e:
                    print(f"  ⚠️  Text attention error: {e}")
                try:
                    visualize_attention_overlay(
                        model_viz, test_loader, answer_vocab, question_vocab,
                        device, n=3, save_path="fig_attention_spatial_overlay.png",
                    )
                except Exception as e:
                    print(f"  ⚠️  Spatial attention error: {e}")
                model_viz.cpu()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    print("\n✓ Visualization complete.")
except Exception as e:
    print(f"⚠️  Visualization error: {e}")

# ═══════════════════════════════════════════════════════════════
# STEP 6: SAVE DEPLOY ARTIFACT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"💾 SAVING DEPLOY ARTIFACT")
print(f"{'='*60}\n")

model_states = {}
for name in cfg.model_variants:
    ckpt_path = os.path.join(cfg.ckpt_dir, f"best_{name}.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_states[name] = checkpoint.get("model", checkpoint)
        print(f"✓ Loaded: {name}")

ensemble_path = os.path.join(project_path, "vqa_deploy_all_models.pth")
torch.save({
    "config": cfg.to_dict(),
    "model_states": model_states,
    "q_vocab": question_vocab,
    "a_vocab": answer_vocab,
}, ensemble_path)
print(f"✓ Saved: {ensemble_path}")
print(f"  Keys: config, model_states ({len(model_states)} models), q_vocab, a_vocab")

print(f"\n{'='*60}")
print(f"🎉 ALL DONE — Total time: {(time.time()-train_start)/3600:.1f} hours")
print(f"{'='*60}")
