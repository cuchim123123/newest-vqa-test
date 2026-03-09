"""
GloVe pretrained word-embedding loader.

Downloads GloVe 6B vectors from Stanford NLP (if not cached), builds an
embedding matrix aligned with a given Vocabulary, and caches the result
for instant subsequent loads.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import Optional

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════
GLOVE_NAME = "6B"
GLOVE_URL = f"https://nlp.stanford.edu/data/glove.{GLOVE_NAME}.zip"
GLOVE_CACHE_DIR = os.path.join("data", "glove_cache")


def download_glove(cache_dir: str = GLOVE_CACHE_DIR) -> str:
    """Download and extract GloVe 6B if not already present.

    Returns:
        Path to the cache directory containing the extracted .txt files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, f"glove.{GLOVE_NAME}.zip")

    # Check if any GloVe txt exists already
    sample_file = os.path.join(cache_dir, f"glove.{GLOVE_NAME}.300d.txt")
    if os.path.exists(sample_file):
        return cache_dir

    # Download (with corruption check — expect ~862 MB for 6B)
    MIN_ZIP_SIZE = 800_000_000  # bytes
    if os.path.exists(zip_path) and os.path.getsize(zip_path) < MIN_ZIP_SIZE:
        print(f"⚠️  Corrupt GloVe zip detected ({os.path.getsize(zip_path):,} bytes). Re-downloading...")
        os.remove(zip_path)

    if not os.path.exists(zip_path):
        print(f"Downloading GloVe {GLOVE_NAME} from {GLOVE_URL} ...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print("Download complete.")

    # Extract
    print(f"Extracting GloVe {GLOVE_NAME} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    print("Extraction complete.")

    return cache_dir


def load_glove_embeddings(
    vocab,
    embed_dim: int = 300,
    cache_dir: str = GLOVE_CACHE_DIR,
) -> Optional[torch.Tensor]:
    """Build an embedding matrix from GloVe vectors for a Vocabulary.

    Args:
        vocab: ``Vocabulary`` instance with ``stoi`` dict mapping.
        embed_dim: Embedding dimension (must match GloVe variant).
        cache_dir: Directory where GloVe files are cached.

    Returns:
        ``(vocab_size, embed_dim)`` float tensor, or ``None`` on failure.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # ── Check for pre-built cached matrix ──────────────────────────
    matrix_path = os.path.join(cache_dir, f"emb_{len(vocab)}_{embed_dim}.pt")
    if os.path.exists(matrix_path):
        print(f"Loading cached embedding matrix from {matrix_path}")
        return torch.load(matrix_path, weights_only=True)

    # ── Load raw GloVe vectors ─────────────────────────────────────
    glove_file = os.path.join(cache_dir, f"glove.{GLOVE_NAME}.{embed_dim}d.txt")
    if not os.path.exists(glove_file):
        print(f"GloVe file not found at {glove_file}. Run download_glove() first.")
        return None

    print(f"Parsing GloVe {GLOVE_NAME} {embed_dim}d vectors ...")
    glove_vectors: dict[str, np.ndarray] = {}
    with open(glove_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] == embed_dim:
                    glove_vectors[word] = vec
            except ValueError:
                continue
    print(f"Loaded {len(glove_vectors):,} GloVe vectors.")

    # ── Build embedding matrix ─────────────────────────────────────
    vocab_size = len(vocab)
    # Xavier-uniform init for words not in GloVe
    scale = np.sqrt(6.0 / (vocab_size + embed_dim))
    embedding_matrix = torch.FloatTensor(vocab_size, embed_dim).uniform_(-scale, scale)

    # Zero for <PAD>
    from src.data.dataset import PAD_IDX
    embedding_matrix[PAD_IDX] = torch.zeros(embed_dim)

    found = 0
    for word, idx in vocab.stoi.items():
        if word in glove_vectors:
            embedding_matrix[idx] = torch.from_numpy(glove_vectors[word])
            found += 1

    coverage = found / vocab_size * 100
    print(f"GloVe coverage: {found}/{vocab_size} words ({coverage:.1f}%)")

    # Cache for next time
    torch.save(embedding_matrix, matrix_path)
    print(f"Cached embedding matrix → {matrix_path}")

    return embedding_matrix
