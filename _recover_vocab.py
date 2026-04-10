"""
Recover the original vocabulary from checkpoint embedding weights + GloVe vectors.

Strategy:
  For each row in the checkpoint's embedding matrix, find the GloVe word whose
  vector is an exact match (cosine similarity ~1.0). Special tokens and OOV words
  (which got Xavier-init, not GloVe vectors) are handled separately.

  We do this for BOTH question vocab (from question_encoder.embedding.weight)
  and answer vocab (from answer_decoder.embedding.weight).
"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, ".")

GLOVE_FILE = "data/glove_cache/glove.6B.300d.txt"
CHECKPOINT_DIR = r"f:\Desktop\vqa\checkpoints"
# Use M1 checkpoint (any would work — they all share same vocab)
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_M1_Scratch_NoAttn.pth")

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


def reverse_map_embedding(emb_weight: torch.Tensor, glove_vectors: dict, name: str) -> dict:
    """
    Given an embedding weight matrix (V, 300) and GloVe vectors,
    recover which word each row corresponds to.
    
    Returns: {index: word} mapping
    """
    V, D = emb_weight.shape
    print(f"\n{'='*60}")
    print(f"Reverse-mapping {name}: {V} embeddings × {D}d")
    print(f"{'='*60}")
    
    # Convert GloVe to matrix for batch comparison
    glove_words = list(glove_vectors.keys())
    glove_matrix = np.stack([glove_vectors[w] for w in glove_words])  # (G, 300)
    
    # Normalize GloVe matrix for cosine similarity
    glove_norms = np.linalg.norm(glove_matrix, axis=1, keepdims=True)
    glove_norms = np.maximum(glove_norms, 1e-10)
    glove_normed = glove_matrix / glove_norms
    
    emb_np = emb_weight.numpy()  # (V, 300)
    
    result = {}
    unmatched = []
    
    # Special tokens first
    for idx, token in SPECIAL_TOKENS.items():
        if idx < V:
            result[idx] = token
    
    t0 = time.time()
    # Process in batches for speed
    batch_size = 256
    for start in range(4, V, batch_size):  # skip first 4 (special tokens)
        end = min(start + batch_size, V)
        batch = emb_np[start:end]  # (B, 300)
        
        # Normalize batch
        batch_norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch_norms = np.maximum(batch_norms, 1e-10)
        batch_normed = batch / batch_norms
        
        # Cosine similarity: (B, G)
        sims = batch_normed @ glove_normed.T
        
        for i in range(end - start):
            idx = start + i
            best_j = np.argmax(sims[i])
            best_sim = sims[i, best_j]
            
            if best_sim > 0.9999:  # Exact match (accounting for float precision)
                result[idx] = glove_words[best_j]
            else:
                # Not found in GloVe — likely OOV word with Xavier init
                unmatched.append((idx, best_sim, glove_words[best_j]))
        
        if (end - 4) % 1000 < batch_size:
            elapsed = time.time() - t0
            print(f"  Processed {end}/{V} ({elapsed:.1f}s) — {len(result)-4} matched, {len(unmatched)} unmatched")
    
    print(f"\n  Final: {len(result)} matched (including {len(SPECIAL_TOKENS)} special), {len(unmatched)} unmatched")
    
    if unmatched:
        print(f"\n  Top 20 unmatched indices (closest GloVe word):")
        for idx, sim, closest in unmatched[:20]:
            print(f"    idx={idx:5d}: best_sim={sim:.4f}, closest='{closest}'")
    
    return result, unmatched


def main():
    # Load GloVe
    glove = load_glove(GLOVE_FILE)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    
    q_emb = sd["question_encoder.embedding.weight"]
    a_emb = sd["answer_decoder.embedding.weight"]
    print(f"  Question embedding: {q_emb.shape}")
    print(f"  Answer embedding:   {a_emb.shape}")
    
    # Reverse map
    q_vocab_map, q_unmatched = reverse_map_embedding(q_emb, glove, "Question Vocab")
    a_vocab_map, a_unmatched = reverse_map_embedding(a_emb, glove, "Answer Vocab")
    
    # Build Vocabulary objects
    from src.data.dataset import Vocabulary
    
    q_vocab = Vocabulary(freq_threshold=1)
    # Clear default special tokens (they're already in our map)
    q_vocab.stoi = {}
    q_vocab.itos = {}
    for idx in sorted(q_vocab_map.keys()):
        word = q_vocab_map[idx]
        q_vocab.stoi[word] = idx
        q_vocab.itos[idx] = word
    # Add placeholder for unmatched
    for idx, sim, closest in q_unmatched:
        word = f"<OOV_q{idx}>"
        q_vocab.stoi[word] = idx
        q_vocab.itos[idx] = word
    
    a_vocab = Vocabulary(freq_threshold=1)
    a_vocab.stoi = {}
    a_vocab.itos = {}
    for idx in sorted(a_vocab_map.keys()):
        word = a_vocab_map[idx]
        a_vocab.stoi[word] = idx
        a_vocab.itos[idx] = word
    for idx, sim, closest in a_unmatched:
        word = f"<OOV_a{idx}>"
        a_vocab.stoi[word] = idx
        a_vocab.itos[idx] = word
    
    print(f"\n{'='*60}")
    print(f"RECOVERED VOCAB SUMMARY")
    print(f"{'='*60}")
    print(f"Question vocab: {len(q_vocab)} words ({len(q_unmatched)} OOV)")
    print(f"Answer vocab:   {len(a_vocab)} words ({len(a_unmatched)} OOV)")
    
    # Save
    out_path = "data/processed/vocab_recovered.pth"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "question_vocab": q_vocab,
        "answer_vocab": a_vocab,
    }, out_path)
    print(f"\nSaved recovered vocab → {out_path}")
    
    # Verify: show first 20 of each
    print(f"\nQuestion vocab first 20:")
    for i in range(min(20, len(q_vocab))):
        print(f"  {i:4d} → {q_vocab.itos[i]}")
    
    print(f"\nAnswer vocab first 20:")
    for i in range(min(20, len(a_vocab))):
        print(f"  {i:4d} → {a_vocab.itos[i]}")
    
    # Verify common words
    print(f"\nSpot checks:")
    for word in ["what", "color", "is", "the", "red", "blue", "green", "yes", "no"]:
        qi = q_vocab.stoi.get(word, "MISSING")
        ai = a_vocab.stoi.get(word, "MISSING")
        print(f"  '{word}': q_idx={qi}, a_idx={ai}")


if __name__ == "__main__":
    main()
