"""
Recover vocab V3 — FAST version using batch matrix ops.
Uses cached (pre-training) GloVe embedding matrices to reverse-map words.
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, ".")

GLOVE_FILE = "data/glove_cache/glove.6B.300d.txt"
Q_CACHED_EMB = "data/glove_cache/emb_2717_300.pt"
A_CACHED_EMB = "data/glove_cache/emb_4203_300.pt"
SPECIAL_TOKENS = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}


def load_glove(path):
    print(f"Loading GloVe from {path}...", flush=True)
    words, vecs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  ... {i:,} lines read", flush=True)
            parts = line.rstrip().split(" ")
            if len(parts) != 301:
                continue
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
    print(f"  Stacking into matrix...", flush=True)
    mat = np.stack(vecs)  # (400K, 300)
    print(f"  Loaded {len(words):,} vectors")
    return words, mat


def reverse_map_fast(cached_emb, glove_words, glove_mat, name):
    V = cached_emb.shape[0]
    emb_np = cached_emb.numpy()  # (V, 300)
    print(f"\n{'='*50}")
    print(f"{name}: {V} rows", flush=True)
    
    t0 = time.time()
    glove_sq = np.sum(glove_mat ** 2, axis=1)  # (G,)
    
    # Process in batches to avoid OOM
    BATCH = 500
    best_j = np.zeros(V, dtype=np.int64)
    best_dist = np.zeros(V, dtype=np.float32)
    
    for start in range(0, V, BATCH):
        end = min(start + BATCH, V)
        batch = emb_np[start:end]  # (B, 300)
        batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dot = batch @ glove_mat.T                              # (B, G)
        d2 = batch_sq + glove_sq[np.newaxis, :] - 2 * dot     # (B, G)
        bj = np.argmin(d2, axis=1)
        bd = np.sqrt(np.maximum(d2[np.arange(end - start), bj], 0))
        best_j[start:end] = bj
        best_dist[start:end] = bd
        print(f"  [{end}/{V}] {time.time()-t0:.1f}s", flush=True)
    
    elapsed = time.time() - t0
    print(f"  Matrix ops done in {elapsed:.1f}s")
    
    result = {}
    unmatched = []
    for idx in range(V):
        if idx in SPECIAL_TOKENS:
            result[idx] = SPECIAL_TOKENS[idx]
        elif best_dist[idx] < 0.01:
            result[idx] = glove_words[best_j[idx]]
        else:
            unmatched.append((idx, float(best_dist[idx]), glove_words[best_j[idx]]))
    
    print(f"  Matched: {len(result)-4} words + 4 special")
    print(f"  Unmatched (OOV): {len(unmatched)}")
    if unmatched:
        print(f"  Sample unmatched:")
        for idx, dist, closest in unmatched[:20]:
            print(f"    idx={idx:5d}: dist={dist:.4f}, closest='{closest}'")
    
    return result, unmatched


def main():
    glove_words, glove_mat = load_glove(GLOVE_FILE)
    
    q_cached = torch.load(Q_CACHED_EMB, map_location="cpu", weights_only=True)
    a_cached = torch.load(A_CACHED_EMB, map_location="cpu", weights_only=True)
    print(f"Q cached: {q_cached.shape}, A cached: {a_cached.shape}")
    
    q_map, q_un = reverse_map_fast(q_cached, glove_words, glove_mat, "Question Vocab")
    a_map, a_un = reverse_map_fast(a_cached, glove_words, glove_mat, "Answer Vocab")
    
    # Show first 30
    for label, m, total in [("Question", q_map, q_cached.shape[0]), ("Answer", a_map, a_cached.shape[0])]:
        print(f"\n{label} vocab (first 30):")
        for i in range(min(30, total)):
            print(f"  {i:4d} → '{m.get(i, '???')}'")
    
    # Spot checks
    q_stoi = {v: k for k, v in q_map.items()}
    a_stoi = {v: k for k, v in a_map.items()}
    print(f"\nSpot checks:")
    for w in ["what","color","is","the","red","blue","green","yes","no","white","black","2","3"]:
        print(f"  '{w}': q={q_stoi.get(w,'MISS')}, a={a_stoi.get(w,'MISS')}")
    
    # Build final Vocabulary objects and save
    from src.data.dataset import Vocabulary
    
    def build_vocab_obj(word_map, unmatched_list, prefix):
        v = Vocabulary(freq_threshold=1)
        v.stoi, v.itos = {}, {}
        for idx in sorted(word_map.keys()):
            w = word_map[idx]
            v.stoi[w] = idx
            v.itos[idx] = w
        for idx, dist, closest in unmatched_list:
            w = f"<OOV_{prefix}{idx}>"
            v.stoi[w] = idx
            v.itos[idx] = w
        return v
    
    q_vocab = build_vocab_obj(q_map, q_un, "q")
    a_vocab = build_vocab_obj(a_map, a_un, "a")
    
    out = "data/processed/vocab_recovered.pth"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"question_vocab": q_vocab, "answer_vocab": a_vocab}, out)
    print(f"\nSaved → {out}  (Q={len(q_vocab)}, A={len(a_vocab)})")


if __name__ == "__main__":
    main()
