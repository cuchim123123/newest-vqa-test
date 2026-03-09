"""Evaluation metrics for VQA: Accuracy, EM, F1, BLEU-1~4, METEOR, WUPS, Top-K, and Semantic Score."""

from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

# ── Semantic Score (BERT) — lazy-loaded on first use ──────────────
_semantic_model = None
_semantic_model_loaded = False

def _get_semantic_model():
    """Lazy-load SentenceTransformer to avoid 80MB download at import time."""
    global _semantic_model, _semantic_model_loaded
    if not _semantic_model_loaded:
        _semantic_model_loaded = True
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            _semantic_model = None
            print(f"Warning: Could not load SentenceTransformer. Semantic score will be 0. Error: {e}")
    return _semantic_model

# ── WUPS (Wu-Palmer Similarity via WordNet) ────────────────────────
try:
    from nltk.corpus import wordnet as wn
    _wn_available = True
except Exception:
    _wn_available = False
    print("Warning: WordNet not available. WUPS will be 0.")

from src.data.preprocessing import normalize_answer, majority_answer


# ════════════════════════════════════════════════════════════════════
# Core metrics
# ════════════════════════════════════════════════════════════════════

def compute_exact_match(pred: str, refs) -> float:
    """Exact match — MAX over multiple refs."""
    if isinstance(refs, str): refs = [refs]
    return float(any(normalize_answer(pred) == normalize_answer(r) for r in refs))

def compute_f1(pred: str, refs) -> float:
    """Token-level F1 — MAX over multiple refs."""
    if isinstance(refs, str): refs = [refs]
    best_f1 = 0.0
    p_toks = normalize_answer(pred).split()
    for r in refs:
        r_toks = normalize_answer(r).split()
        if not p_toks or not r_toks:
            f1 = float(p_toks == r_toks)
        else:
            common = Counter(p_toks) & Counter(r_toks)
            num_same = sum(common.values())
            if num_same == 0:
                f1 = 0.0
            else:
                precision = num_same / len(p_toks)
                recall = num_same / len(r_toks)
                f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

def compute_bleu(pred: str, refs) -> dict[str, float]:
    """BLEU 1-4 with smoothing."""
    if isinstance(refs, str): refs = [refs]
    smoothie = SmoothingFunction().method4
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]

    if not p_toks or not r_toks_list:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    weights = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    return {
        f"bleu{i+1}": sentence_bleu(r_toks_list, p_toks, weights=w, smoothing_function=smoothie)
        for i, w in enumerate(weights)
    }

def compute_meteor(pred: str, refs) -> float:
    """METEOR score (supports N refs)."""
    if isinstance(refs, str): refs = [refs]
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]
    if not p_toks or not r_toks_list:
        return 0.0
    return _nltk_meteor(r_toks_list, p_toks)

def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """VQA soft accuracy: min(#annotators_agreeing / 3, 1.0)."""
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)
    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)


# ════════════════════════════════════════════════════════════════════
# NEW — WUPS@θ (Wu-Palmer Similarity)
# ════════════════════════════════════════════════════════════════════

def _wup_similarity(word_a: str, word_b: str) -> float:
    """Wu-Palmer Similarity between two words using WordNet."""
    if not _wn_available:
        return 0.0
    if word_a == word_b:
        return 1.0
    synsets_a = wn.synsets(word_a)
    synsets_b = wn.synsets(word_b)
    if not synsets_a or not synsets_b:
        return 0.0
    best = 0.0
    for sa in synsets_a:
        for sb in synsets_b:
            score = sa.wup_similarity(sb)
            if score is not None and score > best:
                best = score
    return best


def compute_wups(pred: str, refs, threshold: float = 0.9) -> float:
    """WUPS@θ — Wu-Palmer Similarity thresholded at θ.

    Computes average WUPS over all tokens in the prediction,
    matching each pred token to its best-matching ref token.
    Returns the MAX score over all reference answers.
    """
    if isinstance(refs, str):
        refs = [refs]
    pred_words = normalize_answer(pred).split()
    if not pred_words:
        return 0.0

    best = 0.0
    for ref in refs:
        ref_words = normalize_answer(ref).split()
        if not ref_words:
            continue
        # For each pred token, find best matching ref token
        token_scores = []
        for pw in pred_words:
            best_tok = 0.0
            for rw in ref_words:
                wup = _wup_similarity(pw, rw)
                score = wup if wup >= threshold else threshold * wup
                best_tok = max(best_tok, score)
            token_scores.append(best_tok)
        avg_score = sum(token_scores) / len(token_scores) if token_scores else 0.0
        best = max(best, avg_score)
    return best


# ════════════════════════════════════════════════════════════════════
# NEW — Top-K Classification Accuracy
# ════════════════════════════════════════════════════════════════════

def compute_topk_accuracy(
    cls_logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Top-K accuracy for the multi-task classification head.

    Args:
        cls_logits: ``(B, num_answers)`` raw logits.
        targets:    ``(B,)`` ground-truth class indices.
        k:          Top-K.

    Returns:
        Fraction of samples where the true class is in the top-K predictions.
    """
    if cls_logits is None or targets is None:
        return 0.0
    with torch.no_grad():
        maxk = min(k, cls_logits.size(1))
        _, pred_topk = cls_logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred_topk.eq(targets.unsqueeze(1).expand_as(pred_topk))
        return float(correct.any(dim=1).float().mean().item())


# ════════════════════════════════════════════════════════════════════
# Semantic Score (BERT cosine similarity)
# ════════════════════════════════════════════════════════════════════

def compute_semantic_score(preds: list[str], refs: list) -> float:
    """Semantic similarity via Sentence-BERT cosine."""
    model = _get_semantic_model()
    if model is None or not preds or not refs:
        return 0.0
    from sentence_transformers import util
    clean_preds = [normalize_answer(p) for p in preds]
    clean_refs = [majority_answer(r) if isinstance(r, list) else normalize_answer(r) for r in refs]

    pred_embs = model.encode(clean_preds, convert_to_tensor=True, show_progress_bar=False)
    ref_embs = model.encode(clean_refs, convert_to_tensor=True, show_progress_bar=False)

    cosine_scores = util.cos_sim(pred_embs, ref_embs)
    scores = torch.diag(cosine_scores)
    return float(scores.mean().item())


# ════════════════════════════════════════════════════════════════════
# Batch aggregation
# ════════════════════════════════════════════════════════════════════

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """Aggregate all metrics over a batch of predictions."""
    results: dict[str, list[float]] = {
        "accuracy": [], "em": [], "f1": [], "meteor": [],
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [],
        "wups": [],
    }

    for pred, ref in zip(predictions, references):
        results["accuracy"].append(compute_vqa_accuracy(pred, ref))
        results["em"].append(compute_exact_match(pred, ref))
        results["f1"].append(compute_f1(pred, ref))
        results["meteor"].append(compute_meteor(pred, ref))
        results["wups"].append(compute_wups(pred, ref, threshold=0.9))

        bleus = compute_bleu(pred, ref)
        for k, v in bleus.items():
            results[k].append(v)

    final_metrics = {k: float(np.mean(v)) for k, v in results.items()}
    final_metrics["semantic"] = compute_semantic_score(predictions, references)
    return final_metrics