"""
Reproduce the EXACT vocab that was built during Kaggle training.

This replicates the vocab-building section of run_full_training.py exactly:
  - seed=42, random.shuffle(indices)
  - MAX_TRAIN=15000, MAX_VAL=2500, MAX_TEST=1070
  - train_ratio=0.85
  - freq_threshold=3
  - Uses extract_answer (first rationale > majority direct_answer > choice)
  - Builds question_vocab from questions, answer_vocab from answers
  - Uses train+val indices from hf_train, test indices from hf_val

If the HuggingFace dataset row order hasn't changed, this will produce
the EXACT same vocab (Q=2717, A=4203) with identical word↔index mapping.
"""
import sys, os, random, gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from src.data.preprocessing import extract_answer
from src.data.dataset import Vocabulary

# ── Config (matching run_full_training.py exactly) ─────────────
SEED = 42
FREQ_THRESHOLD = 3
TRAIN_RATIO = 0.85
MAX_TRAIN = 15000
MAX_VAL = 2500
MAX_TEST = 1070
HF_ID = "HuggingFaceM4/A-OKVQA"
OUTPUT = "data/processed/vocab.pth"

# ── Set seed ───────────────────────────────────────────────────
random.seed(SEED)

# ── Load dataset ───────────────────────────────────────────────
print(f"Loading dataset: {HF_ID}...")
hf_train = load_dataset(HF_ID, split="train")
hf_val = load_dataset(HF_ID, split="validation")
print(f"Full HF dataset: train={len(hf_train)}, val={len(hf_val)}")

# ── Subsample indices (exact same logic as run_full_training.py) ──
n_total = len(hf_train)
indices = list(range(n_total))
random.shuffle(indices)

split_idx = int(min(n_total, MAX_TRAIN + MAX_VAL) * TRAIN_RATIO)
train_indices = indices[:min(split_idx, MAX_TRAIN)]
val_indices = indices[split_idx:split_idx + MAX_VAL]
test_indices = list(range(min(len(hf_val), MAX_TEST)))

print(f"Subsampled: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

# ── Build vocab (exact same logic) ────────────────────────────
text_cols = [c for c in hf_train.column_names if c != "image"]
all_questions, all_answers = [], []

for idx in tqdm(train_indices + val_indices, desc="Vocab (train+val)", leave=True, mininterval=1.0):
    row = hf_train.select_columns(text_cols)[idx]
    all_questions.append(row["question"])
    all_answers.append(extract_answer(row))

for idx in tqdm(test_indices, desc="Vocab (test)", leave=True, mininterval=1.0):
    row = hf_val.select_columns(text_cols)[idx]
    all_questions.append(row["question"])
    all_answers.append(extract_answer(row))

question_vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
question_vocab.build_vocabulary(all_questions)
answer_vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
answer_vocab.build_vocabulary(all_answers)
del all_questions, all_answers
gc.collect()

print(f"\n{'='*50}")
print(f"Question Vocab: {len(question_vocab):,}")
print(f"Answer Vocab:   {len(answer_vocab):,}")
print(f"{'='*50}")

# ── Check if sizes match expected ──────────────────────────────
q_expected, a_expected = 2717, 4203
q_ok = len(question_vocab) == q_expected
a_ok = len(answer_vocab) == a_expected
print(f"Q match ({q_expected}): {'✓ YES' if q_ok else '✗ NO — ' + str(len(question_vocab))}")
print(f"A match ({a_expected}): {'✓ YES' if a_ok else '✗ NO — ' + str(len(answer_vocab))}")

if q_ok and a_ok:
    # ── Verify against cached GloVe embedding ─────────────────
    print("\nVerifying against cached GloVe embedding (emb_2717_300.pt)...")
    from src.data.glove import load_glove_embeddings
    q_emb_fresh = load_glove_embeddings(question_vocab, 300)
    q_emb_cached = torch.load("data/glove_cache/emb_2717_300.pt", map_location="cpu", weights_only=True)
    
    if q_emb_fresh is not None:
        diff = (q_emb_fresh - q_emb_cached).abs().max().item()
        print(f"  Max abs diff between fresh and cached Q embedding: {diff:.10f}")
        if diff < 1e-6:
            print("  ✓ PERFECT MATCH — vocab is identical to original training!")
        else:
            print("  ✗ MISMATCH — word order differs from original training")
    
    # Same for answer
    a_emb_fresh = load_glove_embeddings(answer_vocab, 300)
    a_emb_cached = torch.load("data/glove_cache/emb_4203_300.pt", map_location="cpu", weights_only=True)
    
    if a_emb_fresh is not None:
        diff = (a_emb_fresh - a_emb_cached).abs().max().item()
        print(f"  Max abs diff between fresh and cached A embedding: {diff:.10f}")
        if diff < 1e-6:
            print("  ✓ PERFECT MATCH — vocab is identical to original training!")
        else:
            print("  ✗ MISMATCH — word order differs from original training")

# ── Save ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
torch.save({"question_vocab": question_vocab, "answer_vocab": answer_vocab}, OUTPUT)
print(f"\nSaved → {OUTPUT}")

# ── Show sample ────────────────────────────────────────────────
print(f"\nQ vocab first 15:")
for i in range(min(15, len(question_vocab))):
    print(f"  {i:4d} → '{question_vocab.itos[i]}'")
print(f"\nA vocab first 15:")
for i in range(min(15, len(answer_vocab))):
    print(f"  {i:4d} → '{answer_vocab.itos[i]}'")

# Spot checks
print(f"\nSpot checks:")
for w in ["what", "color", "is", "the", "red", "blue", "green", "yes", "no", "white", "black"]:
    qi = question_vocab.stoi.get(w, "MISS")
    ai = answer_vocab.stoi.get(w, "MISS")
    print(f"  '{w}': q={qi}, a={ai}")
