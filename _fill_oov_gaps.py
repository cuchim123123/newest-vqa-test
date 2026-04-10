"""
Fill in the OOV gaps: words that were in the original vocab but NOT in GloVe.
These got Xavier-init embeddings, so we can't reverse-map them from GloVe.

Strategy:
1. Rebuild vocab from the dataset (same code as training)
2. Find words NOT in GloVe (these are the OOV candidates)
3. The recovered vocab has gaps at certain indices — fill them with OOV words
4. Since OOV words got Xavier random init, we can't determine exact mapping
   BUT: we can match them by comparing the cached embedding rows with the
   Xavier-init pattern, or simply try all permutations... 
   
   Actually, simpler: the Xavier init used a SPECIFIC seed (set_seed in training).
   Or even simpler: the OOV words were added in ORDER of first appearance in dataset.
   If we rebuild the same dataset and extract OOV words in order, they should match!

Key insight: `build_vocabulary` iterates `_counter.items()` — in Python 3.7+ this
preserves insertion order. So OOV words appear in the order they were first seen
in the dataset, filtered by freq_threshold.
"""
import sys, os
import torch
import numpy as np

sys.path.insert(0, ".")

GLOVE_FILE = "data/glove_cache/glove.6B.300d.txt"

# Load recovered vocab (with OOV gaps)
recovery = torch.load("data/processed/vocab_recovered.pth", map_location="cpu", weights_only=False)
q_vocab = recovery["question_vocab"]
a_vocab = recovery["answer_vocab"]

# Find gap indices
q_gaps = sorted([i for i in range(len(q_vocab)) if q_vocab.itos[i].startswith("<OOV_")])
a_gaps = sorted([i for i in range(len(a_vocab)) if a_vocab.itos[i].startswith("<OOV_")])
print(f"Q gaps: {len(q_gaps)} indices")
print(f"A gaps: {len(a_gaps)} indices")

# Load GloVe word set (just words, not vectors)
print(f"Loading GloVe word set...", flush=True)
glove_words = set()
with open(GLOVE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split(" ", 1)
        glove_words.add(parts[0])
print(f"  {len(glove_words):,} GloVe words")

# Now rebuild vocab from dataset to get OOV words in order
print(f"\nRebuilding vocab from dataset...", flush=True)
from datasets import load_dataset
from src.data.preprocessing import normalize_answer, tokenize, extract_answer, extract_all_references

ds = load_dataset("HuggingFaceM4/A-OKVQA")
print(f"  Dataset loaded: {list(ds.keys())}")

# Reproduce build_vocabulary logic: count all words, keep freq >= 3
from collections import Counter

q_counter = Counter()
a_counter = Counter()

for split in ["train", "validation"]:
    for item in ds[split]:
        # Questions
        q_text = item.get("question", "")
        for tok in tokenize(q_text):
            q_counter[tok] += 1
        
        # Answers
        answer = extract_answer(item)
        for tok in tokenize(answer):
            a_counter[tok] += 1
        
        # References (rationales + choices)
        for ref in extract_all_references(item):
            for tok in tokenize(ref):
                a_counter[tok] += 1

FREQ_THRESHOLD = 3
q_all_words = [w for w, c in q_counter.items() if c >= FREQ_THRESHOLD]
a_all_words = [w for w, c in a_counter.items() if c >= FREQ_THRESHOLD]

# Find words NOT in GloVe
q_oov = [w for w in q_all_words if w not in glove_words]
a_oov = [w for w in a_all_words if w not in glove_words]

print(f"\nQ words total: {len(q_all_words)}, OOV (not in GloVe): {len(q_oov)}")
print(f"A words total: {len(a_all_words)}, OOV (not in GloVe): {len(a_oov)}")

# Already-recovered non-special, non-OOV words
q_recovered_words = {q_vocab.itos[i] for i in range(4, len(q_vocab)) if not q_vocab.itos[i].startswith("<OOV_")}
a_recovered_words = {a_vocab.itos[i] for i in range(4, len(a_vocab)) if not a_vocab.itos[i].startswith("<OOV_")}

# Filter OOV to only words not already recovered
q_oov_new = [w for w in q_oov if w not in q_recovered_words]
a_oov_new = [w for w in a_oov if w not in a_recovered_words]

print(f"\nQ OOV not yet in vocab: {len(q_oov_new)} (need to fill {len(q_gaps)} gaps)")
print(f"A OOV not yet in vocab: {len(a_oov_new)} (need to fill {len(a_gaps)} gaps)")

print(f"\nQ OOV words (first 50): {q_oov_new[:50]}")
print(f"A OOV words (first 50): {a_oov_new[:50]}")

if len(q_oov_new) == len(q_gaps):
    print(f"\n✅ PERFECT MATCH for Q: {len(q_oov_new)} OOV words = {len(q_gaps)} gaps")
    for gap_idx, word in zip(q_gaps, q_oov_new):
        q_vocab.stoi.pop(q_vocab.itos[gap_idx], None)
        q_vocab.itos[gap_idx] = word
        q_vocab.stoi[word] = gap_idx
else:
    print(f"\n⚠️  Q mismatch: {len(q_oov_new)} OOV words vs {len(q_gaps)} gaps")

if len(a_oov_new) == len(a_gaps):
    print(f"✅ PERFECT MATCH for A: {len(a_oov_new)} OOV words = {len(a_gaps)} gaps")
    for gap_idx, word in zip(a_gaps, a_oov_new):
        a_vocab.stoi.pop(a_vocab.itos[gap_idx], None)
        a_vocab.itos[gap_idx] = word
        a_vocab.stoi[word] = gap_idx
else:
    print(f"⚠️  A mismatch: {len(a_oov_new)} OOV words vs {len(a_gaps)} gaps")

# Spot check
print(f"\nFinal spot checks:")
for w in ["what","color","is","the","red","blue","green","yes","no","white","black","yellow","2","3"]:
    qi = q_vocab.stoi.get(w, "MISS")
    ai = a_vocab.stoi.get(w, "MISS")
    print(f"  '{w}': q={qi}, a={ai}")

# Save final
out = "data/processed/vocab_recovered.pth"
torch.save({"question_vocab": q_vocab, "answer_vocab": a_vocab}, out)
print(f"\nSaved final vocab → {out}")
