"""Export a single deployment artifact from trained checkpoints + vocab.

Packages all model state_dicts, config, and vocabularies into one .pth file
that the web server can load directly.

Usage:
    python scripts/export_deploy_artifact.py
    python scripts/export_deploy_artifact.py --vocab data/processed/vocab.pth --ckpt_dir checkpoints
    python scripts/export_deploy_artifact.py --models M4_Pretrained_Attn M2_Scratch_Attn
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config


def _load_model_state(checkpoint_path: str, model_name: str) -> dict:
    """Load model state_dict from a training checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Invalid checkpoint format for {model_name}: {checkpoint_path}")


def _get_vocab_sizes_from_checkpoint(state_dict: dict) -> tuple[int, int]:
    """Infer vocab sizes from the embedding weight shapes in a state_dict."""
    q_size = state_dict["question_encoder.embedding.weight"].shape[0]
    a_size = state_dict["answer_decoder.embedding.weight"].shape[0]
    return q_size, a_size


def _trim_vocab(vocab, target_size: int):
    """Trim a Vocabulary object to match the target size used during training.

    The rebuilt vocab may have more words than what the model was trained with
    (e.g. HF dataset changed). We keep only the first `target_size` entries
    so the word↔index mapping is consistent for the overlapping portion.
    """
    if len(vocab) == target_size:
        return vocab  # Already matches

    if len(vocab) < target_size:
        # Vocab is smaller than checkpoint — pad with dummy tokens
        for i in range(len(vocab), target_size):
            token = f"<UNUSED_{i}>"
            vocab.stoi[token] = i
            vocab.itos[i] = token
        return vocab

    # Vocab is larger — trim to target_size
    # Keep first target_size entries (special tokens + most frequent words)
    new_stoi = {}
    new_itos = {}
    for idx in range(target_size):
        token = vocab.itos[idx]
        new_stoi[token] = idx
        new_itos[idx] = token
    vocab.stoi = new_stoi
    vocab.itos = new_itos
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VQA deployment artifact")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.pth",
                        help="Path to saved vocab file")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Override checkpoint directory (default: from config)")
    parser.add_argument("--output", type=str, default="vqa_deploy_all_models.pth",
                        help="Output path for deployment artifact")
    parser.add_argument("--models", nargs="+",
                        help="Specific model names to package. Default: all variants from config")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    ckpt_dir = args.ckpt_dir or cfg.ckpt_dir

    # ── Load vocabulary ───────────────────────────────────────────
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab}")

    vocabs = torch.load(args.vocab, map_location="cpu", weights_only=False)
    q_vocab = vocabs.get("q_vocab") or vocabs.get("question_vocab")
    a_vocab = vocabs.get("a_vocab") or vocabs.get("answer_vocab")
    if q_vocab is None or a_vocab is None:
        raise RuntimeError("Vocabulary file must contain 'q_vocab'/'question_vocab' and 'a_vocab'/'answer_vocab'")

    print(f"Loaded vocab: Q={len(q_vocab)}, A={len(a_vocab)}")

    # ── Collect model state dicts ─────────────────────────────────
    selected = args.models or list(cfg.model_variants.keys())
    model_states: dict[str, dict] = {}

    for name in selected:
        if name not in cfg.model_variants:
            print(f"  ⚠ '{name}' not in config model_variants, skipping")
            continue
        ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pth")
        try:
            model_states[name] = _load_model_state(ckpt_path, name)
            print(f"  ✓ Loaded {name} ({len(model_states[name])} params)")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")

    if not model_states:
        raise RuntimeError("No model states collected — nothing to export")

    # ── Detect vocab sizes from checkpoint embeddings ─────────────
    # Use the first checkpoint to determine what vocab size the models expect
    first_state = next(iter(model_states.values()))
    expected_q, expected_a = _get_vocab_sizes_from_checkpoint(first_state)
    print(f"\nCheckpoint expects: Q={expected_q}, A={expected_a}")
    print(f"Loaded vocab has:   Q={len(q_vocab)}, A={len(a_vocab)}")

    if len(q_vocab) != expected_q or len(a_vocab) != expected_a:
        print("⚠ Vocab size mismatch — trimming vocab to match checkpoints...")
        q_vocab = _trim_vocab(q_vocab, expected_q)
        a_vocab = _trim_vocab(a_vocab, expected_a)
        print(f"  Trimmed to: Q={len(q_vocab)}, A={len(a_vocab)}")

    # ── Build clean variant config (strip train_overrides) ────────
    clean_variants = {}
    for name in model_states:
        raw = dict(cfg.model_variants[name])
        raw.pop("train_overrides", None)  # Remove training-only config
        clean_variants[name] = raw

    # ── Package artifact ──────────────────────────────────────────
    artifact = {
        "config": {
            "model": {
                "embed_size": cfg.model.embed_size,
                "hidden_size": cfg.model.hidden_size,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
                "bidirectional": cfg.model.bidirectional,
            },
            "data": {
                "image_size": cfg.data.image_size,
            },
            "model_variants": clean_variants,
        },
        "model_states": model_states,
        "q_vocab": q_vocab,
        "a_vocab": a_vocab,
    }

    torch.save(artifact, args.output)
    print(f"\n✅ Exported deployment artifact to: {args.output}")
    print(f"   Models: {', '.join(model_states.keys())}")
    print(f"   Vocab:  Q={len(q_vocab)}, A={len(a_vocab)}")


if __name__ == "__main__":
    main()
