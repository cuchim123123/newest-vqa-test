"""
Fill answer OOV gaps with the most frequent OOV candidates.
Since we have 209 candidates for 60 gaps, pick the top-60 by frequency.
The exact order within OOV may not be perfect, but these are all rare words
that the model rarely predicts anyway.
"""
import sys, os
import torch
from collections import Counter

sys.path.insert(0, ".")

GLOVE_FILE = "data/glove_cache/glove.6B.300d.txt"

# Load current vocab
recovery = torch.load("data/processed/vocab_recovered.pth", map_location="cpu", weights_only=False)
q_vocab = recovery["question_vocab"]
a_vocab = recovery["answer_vocab"]

# Q is already perfect, only fix A
a_gaps = sorted([i for i in range(len(a_vocab)) if a_vocab.itos[i].startswith("<OOV_")])
print(f"A gaps to fill: {len(a_gaps)}")

# Load GloVe word set
glove_words = set()
with open(GLOVE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        glove_words.add(line.split(" ", 1)[0])

# Rebuild answer counter from dataset
from datasets import load_dataset
from src.data.preprocessing import tokenize, extract_answer, extract_all_references

ds = load_dataset("HuggingFaceM4/A-OKVQA")
a_counter = Counter()
for split in ["train", "validation"]:
    for item in ds[split]:
        answer = extract_answer(item)
        for tok in tokenize(answer):
            a_counter[tok] += 1
        for ref in extract_all_references(item):
            for tok in tokenize(ref):
                a_counter[tok] += 1

# Words already in recovered vocab
a_recovered = {a_vocab.itos[i] for i in range(len(a_vocab)) if not a_vocab.itos[i].startswith("<OOV_")}

# OOV candidates: freq>=3, not in GloVe, not already recovered
a_oov_candidates = [(w, c) for w, c in a_counter.items() 
                     if c >= 3 and w not in glove_words and w not in a_recovered]
# Sort by frequency (most common first)
a_oov_candidates.sort(key=lambda x: -x[1])

print(f"OOV candidates (freq>=3, not in GloVe, not recovered): {len(a_oov_candidates)}")
print(f"Top 70 by frequency:")
for w, c in a_oov_candidates[:70]:
    print(f"  '{w}': freq={c}")

# Take top 60
if len(a_oov_candidates) >= len(a_gaps):
    fill_words = [w for w, c in a_oov_candidates[:len(a_gaps)]]
    print(f"\nFilling {len(a_gaps)} gaps with top-{len(a_gaps)} OOV words")
    for gap_idx, word in zip(a_gaps, fill_words):
        old = a_vocab.itos[gap_idx]
        a_vocab.stoi.pop(old, None)
        a_vocab.itos[gap_idx] = word
        a_vocab.stoi[word] = gap_idx
    print("Done!")
else:
    print(f"Not enough candidates! {len(a_oov_candidates)} < {len(a_gaps)}")

# Final checks
print(f"\nFinal spot checks:")
for w in ["what","color","is","the","red","blue","green","yes","no","white","black","yellow","it's","don't","they're"]:
    qi = q_vocab.stoi.get(w, "MISS")
    ai = a_vocab.stoi.get(w, "MISS")
    print(f"  '{w}': q={qi}, a={ai}")

# Count remaining OOV
remaining = sum(1 for i in range(len(a_vocab)) if a_vocab.itos[i].startswith("<OOV_"))
print(f"\nRemaining OOV in A: {remaining}")
remaining_q = sum(1 for i in range(len(q_vocab)) if q_vocab.itos[i].startswith("<OOV_"))
print(f"Remaining OOV in Q: {remaining_q}")

# Save
out = "data/processed/vocab_recovered.pth"
torch.save({"question_vocab": q_vocab, "answer_vocab": a_vocab}, out)
print(f"\nSaved → {out}")
print(f"Q: {len(q_vocab)}, A: {len(a_vocab)}")
