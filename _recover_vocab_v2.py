"""
Recover vocab V2: Use nearest-neighbor GloVe matching with lower threshold.
The embeddings have been fine-tuned during training, so we can't expect exact match.
We use the closest GloVe word for each embedding row.

Key insight: Since Python 3.7+ dicts maintain insertion order, and the vocab was
built by iterating over the HuggingFace dataset, the word ORDER in vocab depends
on the dataset iteration order at training time. We CANNOT reconstruct this order
from scratch — we must recover it from the embeddings.

Strategy:
1. For each embedding row, find the closest GloVe word
2. For words not in GloVe (compound words, numbers, etc.), we need special handling
3. We must verify by checking the cached GloVe embedding matrix emb_2717_300.pt and
   emb_4203_300.pt — these were built BEFORE training, so they ARE the original GloVe
   vectors for the correct word order!
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, ".")

GLOVE_FILE = "data/glove_cache/glove.6B.300d.txt"

# These cached embedding matrices were built from vocab BEFORE training!
# So they contain the ORIGINAL GloVe vectors in the correct word order!
Q_CACHED_EMB = "data/glove_cache/emb_2717_300.pt"
A_CACHED_EMB = "data/glove_cache/emb_4203_300.pt"

SPECIAL_TOKENS = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}


def load_glove(path: str) -> dict:
    """Load GloVe word→vector mapping."""
    print(f"Loading GloVe from {path}...")
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] == 300:
                    vectors[word] = vec
            except ValueError:
                continue
    print(f"  Loaded {len(vectors):,} GloVe vectors")
    return vectors


def reverse_map_cached_embedding(cached_emb: torch.Tensor, glove_vectors: dict, name: str):
    """
    Match CACHED (pre-training) embedding rows against GloVe vectors.
    These should be EXACT matches since the cache was built by copying GloVe vectors.
    """
    V, D = cached_emb.shape
    print(f"\n{'='*60}")
    print(f"Reverse-mapping {name}: {V} embeddings × {D}d (from CACHED pre-training matrix)")
    print(f"{'='*60}")
    
    # Build GloVe lookup
    glove_words = list(glove_vectors.keys())
    glove_matrix = np.stack([glove_vectors[w] for w in glove_words])  # (G, 300)
    
    emb_np = cached_emb.numpy()
    
    result = {}
    unmatched_indices = []
    
    # Special tokens
    for idx, token in SPECIAL_TOKENS.items():
        if idx < V:
            result[idx] = token
    
    t0 = time.time()
    batch_size = 512
    
    for start in range(4, V, batch_size):
        end = min(start + batch_size, V)
        batch = emb_np[start:end]  # (B, 300)
        
        # Try exact match first (L2 distance)
        # Since cached emb was COPIED from GloVe, L2 should be ~0
        for i in range(end - start):
            idx = start + i
            row = batch[i]
            
            # Check if this is the zero vector (PAD) or Xavier-init (no GloVe match)
            row_norm = np.linalg.norm(row)
            if row_norm < 1e-6:
                unmatched_indices.append((idx, 0.0, "<ZERO>"))
                continue
            
            # Compute L2 distance against all GloVe vectors
            dists = np.linalg.norm(glove_matrix - row[np.newaxis, :], axis=1)
            best_j = np.argmin(dists)
            best_dist = dists[best_j]
            
            if best_dist < 0.001:  # Essentially exact match
                result[idx] = glove_words[best_j]
            else:
                unmatched_indices.append((idx, best_dist, glove_words[best_j]))
        
        elapsed = time.time() - t0
        done = end - 4
        total = V - 4
        if done % 500 < batch_size or end == V:
            print(f"  [{done}/{total}] {elapsed:.1f}s — {len(result)-4} GloVe-matched, {len(unmatched_indices)} unmatched")
    
    print(f"\n  Summary: {len(result)} total ({len(result)-4} words + 4 special), {len(unmatched_indices)} unmatched")
    
    if unmatched_indices:
        print(f"\n  Sample unmatched (these are OOV words with Xavier init):")
        for idx, dist, closest in unmatched_indices[:30]:
            print(f"    idx={idx:5d}: L2_dist={dist:.4f}, closest='{closest}'")
    
    return result, unmatched_indices


def main():
    # Load GloVe
    glove = load_glove(GLOVE_FILE)
    
    # Load CACHED embedding matrices (pre-training, original GloVe vectors)
    print(f"\nLoading cached embeddings...")
    q_cached = torch.load(Q_CACHED_EMB, map_location="cpu", weights_only=True)
    a_cached = torch.load(A_CACHED_EMB, map_location="cpu", weights_only=True)
    print(f"  Q cached: {q_cached.shape}")
    print(f"  A cached: {a_cached.shape}")
    
    # Reverse map using cached (not trained) embeddings
    q_map, q_unmatched = reverse_map_cached_embedding(q_cached, glove, "Question Vocab")
    a_map, a_unmatched = reverse_map_cached_embedding(a_cached, glove, "Answer Vocab")
    
    print(f"\n{'='*60}")
    print(f"RECOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"Question: {len(q_map)}/{q_cached.shape[0]} recovered ({len(q_unmatched)} OOV)")
    print(f"Answer:   {len(a_map)}/{a_cached.shape[0]} recovered ({len(a_unmatched)} OOV)")
    
    # Show first 30
    print(f"\nQuestion vocab first 30:")
    for i in range(min(30, q_cached.shape[0])):
        word = q_map.get(i, "???")
        print(f"  {i:4d} → '{word}'")
    
    print(f"\nAnswer vocab first 30:")
    for i in range(min(30, a_cached.shape[0])):
        word = a_map.get(i, "???")
        print(f"  {i:4d} → '{word}'")
    
    # Spot check
    print(f"\nSpot checks:")
    # Build reverse for checking
    q_stoi = {v: k for k, v in q_map.items()}
    a_stoi = {v: k for k, v in a_map.items()}
    for word in ["what", "color", "is", "the", "red", "blue", "green", "yes", "no", "white", "black", "yellow"]:
        qi = q_stoi.get(word, "MISS")
        ai = a_stoi.get(word, "MISS")
        print(f"  '{word}': q_idx={qi}, a_idx={ai}")
    
    # Save partial result for analysis
    torch.save({
        "q_map": q_map,
        "a_map": a_map,
        "q_unmatched": q_unmatched,
        "a_unmatched": a_unmatched,
    }, "data/processed/_recovery_data.pth")
    print(f"\nSaved recovery data → data/processed/_recovery_data.pth")


if __name__ == "__main__":
    main()
