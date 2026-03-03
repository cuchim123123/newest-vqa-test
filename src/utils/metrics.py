"""Evaluation metrics for VQA: Accuracy, EM, F1, BLEU-1~4, METEOR, and Semantic Score."""

from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

# Thêm import cho Semantic Score
try:
    from sentence_transformers import SentenceTransformer, util
    # Khởi tạo mô hình BERT nhẹ (chạy cực nhanh, tốn rất ít VRAM)
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    semantic_model = None
    print(f"Warning: Could not load SentenceTransformer. Semantic score will be 0. Error: {e}")

# Sử dụng hàm chuẩn hóa mạnh mẽ đã có trong dự án của nhóm
from src.data.preprocessing import normalize_answer, majority_answer

def compute_exact_match(pred: str, refs) -> float:
    """So khớp chính xác lấy MAX (soft match over multiple refs)."""
    if isinstance(refs, str): refs = [refs]
    return float(any(normalize_answer(pred) == normalize_answer(r) for r in refs))

def compute_f1(pred: str, refs) -> float:
    """Tính F1-score ở mức độ token. Lấy MAX over multiple refs."""
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
    """Tính BLEU từ 1 đến 4 sử dụng corpus-level refs."""
    if isinstance(refs, str): refs = [refs]
    smoothie = SmoothingFunction().method4
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]
    
    if not p_toks or not r_toks_list:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    weights = [
        (1, 0, 0, 0),          # BLEU-1
        (0.5, 0.5, 0, 0),      # BLEU-2
        (0.33, 0.33, 0.33, 0), # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    
    return {
        f"bleu{i+1}": sentence_bleu(r_toks_list, p_toks, weights=w, smoothing_function=smoothie)
        for i, w in enumerate(weights)
    }

def compute_meteor(pred: str, refs) -> float:
    """Tính METEOR score (hỗ trợ N refs)."""
    if isinstance(refs, str): refs = [refs]
    p_toks = normalize_answer(pred).split()
    r_toks_list = [normalize_answer(r).split() for r in refs if normalize_answer(r).strip()]
    if not p_toks or not r_toks_list:
        return 0.0
    return _nltk_meteor(r_toks_list, p_toks)

def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """
    Tính VQA Accuracy mềm: min(#người_cùng_đáp_án / 3, 1.0).
    Sử dụng cho các tập dữ liệu có nhiều người gắn nhãn (như A-OKVQA).
    """
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)
    
    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)

def compute_semantic_score(preds: list[str], refs: list) -> float:
    """Tính điểm tương đồng ngữ nghĩa bằng Cosine Similarity."""
    if not semantic_model or not preds or not refs:
        return 0.0
    
    clean_preds = [normalize_answer(p) for p in preds]
    # Lấy the most representative string if it's a list for semantic comparison
    clean_refs = [majority_answer(r) if isinstance(r, list) else normalize_answer(r) for r in refs]
    
    # Mã hóa thành Vector (Embeddings)
    pred_embs = semantic_model.encode(clean_preds, convert_to_tensor=True, show_progress_bar=False)
    ref_embs = semantic_model.encode(clean_refs, convert_to_tensor=True, show_progress_bar=False)
    
    # Tính ma trận độ lệch Cosine và lấy đường chéo (so sánh 1-1)
    cosine_scores = util.cos_sim(pred_embs, ref_embs)
    scores = torch.diag(cosine_scores)
    
    return float(scores.mean().item())

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đo lường trên batch."""
    results = {
        "accuracy": [], "em": [], "f1": [], "meteor": [],
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []
    }
    
    for pred, ref in zip(predictions, references):
        # Truyền toàn bộ list refs cho hàm compute_f1, compute_bleu để tối đa hóa điểm
        results["accuracy"].append(compute_vqa_accuracy(pred, ref))
        results["em"].append(compute_exact_match(pred, ref))
        results["f1"].append(compute_f1(pred, ref))
        results["meteor"].append(compute_meteor(pred, ref))
        
        bleus = compute_bleu(pred, ref)
        for k, v in bleus.items():
            results[k].append(v)
            
    # Tính trung bình các chỉ số truyền thống
    final_metrics = {k: float(np.mean(v)) for k, v in results.items()}
    
    # Tính Semantic Score cho toàn bộ batch
    final_metrics["semantic"] = compute_semantic_score(predictions, references)
    
    return final_metrics