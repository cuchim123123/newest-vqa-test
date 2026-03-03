"""Evaluation pipeline with question-type breakdown."""

from __future__ import annotations
import logging
import os
from collections import defaultdict
from typing import Any
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PAD_IDX
from src.data.preprocessing import majority_answer, classify_question
from src.utils.helpers import decode_sequence
from src.utils.metrics import batch_metrics, compute_exact_match, compute_f1, compute_meteor

logger = logging.getLogger("VQA")

def evaluate_model(
    model: torch.nn.Module, test_loader: DataLoader, answer_vocab: Any,
    question_vocab: Any, device: torch.device, ckpt_dir: str = "checkpoints",
    name: str = "model", beam_width: int = 5
) -> dict[str, Any]:
    
    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"[{name}] Missing keys (will use defaults): {missing}")
        if unexpected:
            logger.warning(f"[{name}] Unexpected keys (ignored): {unexpected}")
        logger.info(f"Loaded best checkpoint for {name}")

    model.eval()
    preds, refs, questions_text = [], [], []

    with torch.no_grad():
        for imgs, qs, ql, ans, al, ans_txt, raw_qs in tqdm(test_loader, desc=f"Test {name}"):
            imgs, qs, ql = imgs.to(device), qs.to(device), ql.to(device)
            gen = model.generate(imgs, qs, ql, use_beam=True, beam_width=beam_width, raw_questions=raw_qs)
            for i in range(gen.size(0)):
                preds.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                refs.append(ans_txt[i])
                questions_text.append(decode_sequence(qs[i].cpu().tolist(), question_vocab))

    m = batch_metrics(preds, refs)
    logger.info(f"  {name} F1={m['f1']:.4f} BLEU1={m['bleu1']:.4f} BLEU4={m['bleu4']:.4f} METEOR={m['meteor']:.4f}")

    return {"metrics": m, "preds": preds, "refs": refs, "questions": questions_text}

def evaluate_by_question_type(preds, refs, questions):
    type_data = defaultdict(lambda: {"preds": [], "refs": []})
    for p, r, q in zip(preds, refs, questions):
        qtype = classify_question(q)
        type_data[qtype]["preds"].append(p)
        type_data[qtype]["refs"].append(r)

    results = {}
    import numpy as np
    for qtype, data in sorted(type_data.items(), key=lambda x: -len(x[1]["preds"])):
        ems = [compute_exact_match(p, r) for p, r in zip(data["preds"], data["refs"])]
        f1s = [compute_f1(p, r) for p, r in zip(data["preds"], data["refs"])]
        results[qtype] = {"total": len(data["preds"]), "em": float(np.mean(ems)), "f1": float(np.mean(f1s))}
    return results

def get_failure_cases(preds, refs, questions, n=20):
    failures = []
    for p, r, q in zip(preds, refs, questions):
        # r is a list of valid references
        f1 = compute_f1(p, r)
        rep_ref = r[0] if isinstance(r, list) else r
        failures.append({"question": q, "prediction": p, "reference": rep_ref, "f1": f1, "type": classify_question(q)})
    failures.sort(key=lambda x: x["f1"])
    return failures[:n]