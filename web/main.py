"""VQA Web Server — FastAPI backend for Visual Question Answering.

Loads a deployment artifact (vqa_deploy_all_models.pth) containing all 4 model
variants and vocabularies, then serves a prediction API + static frontend.

Usage:
    cd DL_VQA_Project
    uvicorn web.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
import sys
from typing import Dict, Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.models.vqa_model import VQAModel
from src.data.preprocessing import normalize_answer
from src.utils.helpers import decode_sequence

# ═══════════════════════════════════════════════════════════════════
# App & State
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(title="VQA System", version="1.0.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class ServerState:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, VQAModel] = {}
        self.q_vocab = None
        self.a_vocab = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


state = ServerState()


# ═══════════════════════════════════════════════════════════════════
# Vocab helpers (support both Vocabulary objects and plain dicts)
# ═══════════════════════════════════════════════════════════════════

def _vocab_len(vocab) -> int:
    if hasattr(vocab, "__len__"):
        return len(vocab)
    if isinstance(vocab, dict):
        for k in ("itos", "stoi"):
            if k in vocab and isinstance(vocab[k], dict):
                return len(vocab[k])
    raise RuntimeError("Cannot determine vocab size")


def _vocab_stoi(vocab) -> dict:
    if hasattr(vocab, "stoi"):
        return vocab.stoi
    if isinstance(vocab, dict) and "stoi" in vocab:
        return vocab["stoi"]
    raise RuntimeError("Vocab has no stoi mapping")


def _numericalize(vocab, text: str) -> list[int]:
    if hasattr(vocab, "numericalize"):
        return vocab.numericalize(text)
    normalized = normalize_answer(text)
    stoi = _vocab_stoi(vocab)
    unk = stoi.get("<UNK>", 1)
    return [stoi.get(tok, unk) for tok in normalized.split()]


# ═══════════════════════════════════════════════════════════════════
# Startup — load artifact
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
def load_models() -> None:
    artifact_path = os.environ.get("VQA_ARTIFACT", "vqa_deploy_all_models.pth")
    if not os.path.exists(artifact_path):
        raise RuntimeError(
            f"Deployment artifact not found: {artifact_path}\n"
            "Run: python scripts/export_deploy_artifact.py"
        )

    print(f"Loading artifact from {artifact_path} ...")
    artifact = torch.load(artifact_path, map_location=state.device, weights_only=False)

    cfg = artifact.get("config", {})
    model_states = artifact.get("model_states", {})
    state.q_vocab = artifact.get("q_vocab")
    state.a_vocab = artifact.get("a_vocab")

    if not model_states or state.q_vocab is None or state.a_vocab is None:
        raise RuntimeError("Artifact is missing required data (model_states, q_vocab, a_vocab)")

    model_cfg = cfg.get("model", {})
    variants = cfg.get("model_variants", {})

    for name, variant in variants.items():
        if name not in model_states:
            print(f"  ⚠ Skipping {name} — no state_dict in artifact")
            continue

        model = VQAModel(
            q_vocab_size=_vocab_len(state.q_vocab),
            a_vocab_size=_vocab_len(state.a_vocab),
            embed_size=model_cfg.get("embed_size", 300),
            hidden_size=model_cfg.get("hidden_size", 512),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            bidirectional=model_cfg.get("bidirectional", True),
            **variant,
        ).to(state.device)

        model.load_state_dict(model_states[name], strict=False)
        model.eval()
        state.models[name] = model
        print(f"  ✓ Loaded {name}")

    if not state.models:
        raise RuntimeError("No models loaded from artifact")
    print(f"Ready — {len(state.models)} models on {state.device}")


# ═══════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.post("/v1/predict")
async def predict(
    question: str = Form(..., description="Question about the image"),
    model_name: Optional[str] = Form(None, description="Model variant (omit = run all)"),
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
) -> JSONResponse:
    if not state.models:
        raise HTTPException(500, "Models not loaded")

    # Parse image
    try:
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Failed to read image file")

    img_tensor = state.transform(pil_img).unsqueeze(0).to(state.device)

    # Tokenize question
    stoi = _vocab_stoi(state.q_vocab)
    sos = stoi.get("<SOS>", 2)
    eos = stoi.get("<EOS>", 3)
    q_tokens = [sos] + _numericalize(state.q_vocab, question) + [eos]
    q_tensor = torch.tensor(q_tokens, dtype=torch.long).unsqueeze(0).to(state.device)
    q_len = torch.tensor([len(q_tokens)], dtype=torch.long)

    # Select models
    if model_name:
        if model_name not in state.models:
            raise HTTPException(400, f"Unknown model: {model_name}")
        targets = {model_name: state.models[model_name]}
    else:
        targets = state.models

    # Inference
    results: Dict[str, str] = {}
    for name, model in targets.items():
        with torch.no_grad():
            gen = model.generate(
                img_tensor, q_tensor, q_len,
                use_beam=True,
                beam_width=8,
                rep_penalty=1.2,
                min_gen_len=3,
                len_alpha=0.6,
                suppress_unk=True,
                temperature=0.9,
            )
        answer = decode_sequence(gen[0].cpu().tolist(), state.a_vocab)
        results[name] = answer

    return JSONResponse({
        "question": question,
        "answers": results,
        "models": list(results.keys()),
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(state.device),
        "models_loaded": list(state.models.keys()),
    }


@app.get("/", include_in_schema=False)
def index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(500, "Frontend not found")
    return FileResponse(index_path)
