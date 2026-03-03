"""Training pipeline with early stopping, checkpointing, and warmup."""

from __future__ import annotations
import logging
import os
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PAD_IDX
from src.utils.helpers import decode_sequence
from src.utils.metrics import batch_metrics

logger = logging.getLogger("VQA")

class EarlyStopping:
    """Dừng huấn luyện nếu F1-score không cải thiện."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def train_model(
    model: nn.Module, name: str, train_loader: DataLoader, val_loader: DataLoader,
    answer_vocab: Any, device: torch.device, epochs: int = 20, lr: float = 3e-4,
    use_beam: bool = False, beam_w: int = 5, ckpt_dir: str = "checkpoints",
    label_smoothing: float = 0.1, patience: int = 5, grad_clip: float = 5.0,
    tf_start: float = 1.0, tf_end: float = 0.5, warmup_epochs: int = 3,
    eval_every: int = 1, use_amp: bool = False
) -> dict[str, list[float]]:
    
    os.makedirs(ckpt_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)
    
    # 🟢 [CẬP NHẬT] Differential Learning Rates
    # Phân tách tham số thành 2 nhóm: Pretrained (ResNet) và Scratch (từ đầu)
    pretrained_params = []
    scratch_params = []
    
    if hasattr(model, 'image_encoder') and getattr(model.image_encoder, 'pretrained', False):
        pretrained_params = list(model.image_encoder.get_pretrained_params())
        
        # Mọi thứ còn lại đưa vào nhóm đổi mới (scratch)
        pretrained_ids = {id(p) for p in pretrained_params}
        scratch_params = [p for p in model.parameters() if id(p) not in pretrained_ids]
        
        logger.info(f"[{name}] Using differential learning rates. Pretrained LR: {lr * 0.1:.1e}, Scratch LR: {lr:.1e}")
        optimizer = optim.Adam([
            {'params': pretrained_params, 'lr': lr * 0.1},
            {'params': scratch_params, 'lr': lr}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Cosine scheduler với Warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=patience)

    # 🟢 [CẬP NHẬT] GradScaler để dùng AMP (chống OOM)
    # Chỉ bật nếu dùng CUDA (T4 GPU)
    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and is_cuda))

    best_f1: float = 0.0
    last_metrics: dict[str, float] = {k: 0.0 for k in ["accuracy", "em", "f1", "meteor", "bleu4"]}
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "val_acc": [], "val_em": [], "val_f1": [], "val_meteor": [],
        "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
    }

    for epoch in range(1, epochs + 1):
        # 1. Warmup & TF ratio calculation
        if epoch <= warmup_epochs:
            for i, g in enumerate(optimizer.param_groups): 
                # Nhóm 0 là Pretrained (nếu có), Nhóm 1 (hoặc 0) là Scratch
                target_lr = lr * 0.1 if (len(optimizer.param_groups) > 1 and i == 0) else lr
                g['lr'] = target_lr * (epoch / warmup_epochs)
        
        # Exponential Decay Teacher Forcing (Mục 2)
        import math
        # tf = tf_start giảm mượt mà và tiến sát về tf_end ở những epoch cuối
        decay_rate = 5.0 / epochs  # Đảm bảo đường cong phù hợp
        tf = max(tf_end, tf_start * math.exp(-decay_rate * (epoch - 1)))

        # ── TRAIN ──
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Ep {epoch}/{epochs} tf={tf:.2f}")

        for imgs, qs, ql, ans, al, _, raw_qs in pbar:
            imgs, qs, ql, ans = imgs.to(device), qs.to(device), ql.to(device), ans.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # 🟢 [CẬP NHẬT] Bọc forward pass trong autocast
            with torch.autocast('cuda', enabled=(use_amp and is_cuda)):
                # Truyền raw_qs cho các mô hình nâng cao (nếu cần)
                out = model(imgs, qs, ql, ans, tf_ratio=tf, raw_questions=raw_qs)
                loss = criterion(out.reshape(-1, out.size(-1)), ans[:, 1:].reshape(-1))
            
            # 🟢 [CẬP NHẬT] Scaling loss trước khi backward
            scaler.scale(loss).backward()
            
            # Unscale trước khi clip grad norm
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Cập nhật optimizer thông qua scaler
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / len(train_loader)

        # ── VALIDATE ──
        do_full_eval = (epoch % eval_every == 0) or (epoch == epochs)
        model.eval()
        val_loss_sum = 0.0
        preds_all, refs_all = [], []

        with torch.no_grad():
            for imgs, qs, ql, ans, al, ans_txt, raw_qs in val_loader:
                imgs, qs, ql, ans = imgs.to(device), qs.to(device), ql.to(device), ans.to(device)
                
                with torch.autocast('cuda', enabled=(use_amp and is_cuda)):
                    out = model(imgs, qs, ql, ans, tf_ratio=0, raw_questions=raw_qs)
                    val_loss_sum += criterion(out.reshape(-1, out.size(-1)), ans[:, 1:].reshape(-1)).item()

                if do_full_eval:
                    gen = model.generate(imgs, qs, ql, use_beam=use_beam, beam_width=beam_w, raw_questions=raw_qs)
                    for i in range(gen.size(0)):
                        preds_all.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                        refs_all.append(ans_txt[i])

        val_loss = val_loss_sum / len(val_loader)
        if do_full_eval:
            m = batch_metrics(preds_all, refs_all)
            last_metrics = m
        else:
            m = last_metrics 

        if epoch > warmup_epochs: scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)
        for k in ["val_acc", "val_em", "val_f1", "val_meteor", "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4"]:
            metric_key = k.replace("val_", "").replace("acc", "accuracy")
            history[k].append(m.get(metric_key, 0.0))

        logger.info(f"  Ep {epoch:>2d} loss={train_loss:.3f}/{val_loss:.3f} F1={m['f1']:.3f} B4={m['bleu4']:.3f} lr={cur_lr:.1e}")

        if do_full_eval and m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_f1": best_f1}, os.path.join(ckpt_dir, f"best_{name}.pth"))
            logger.info(f"    ★ Saved best (F1={best_f1:.4f})")

        if do_full_eval and stopper(m["f1"]):
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    return history