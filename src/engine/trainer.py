"""Training pipeline with early stopping, checkpointing, warmup, and multi-task loss."""

from __future__ import annotations
import logging
import math
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
from src.utils.metrics import batch_metrics, compute_topk_accuracy

logger = logging.getLogger("VQA")

class EarlyStopping:
    """Stop training if F1-score does not improve."""
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
    answer_vocab: Any, device: torch.device, epochs: int = 25, lr: float = 5e-4,
    use_beam: bool = False, beam_w: int = 5, ckpt_dir: str = "checkpoints",
    label_smoothing: float = 0.1, patience: int = 7, grad_clip: float = 1.0,
    tf_start: float = 1.0, tf_end: float = 0.4, warmup_epochs: int = 3,
    eval_every: int = 1, use_amp: bool = False,
    cls_weight: float = 0.0, answer_to_idx: Optional[dict] = None,
    weight_decay: float = 1e-5, pretrained_lr_ratio: float = 0.1,
    unfreeze_after_epoch: int = 999,
) -> dict[str, list[float]]:
    """Train a VQA model with optional multi-task classification.

    Saves a resumable checkpoint (resume_<name>.pth) every epoch so training
    can continue after a crash.  The best-model checkpoint (best_<name>.pth)
    is saved separately whenever F1 improves.

    Args:
        cls_weight:    Weight for the auxiliary classification loss (0 = disabled).
        answer_to_idx: Mapping from answer text → class index for the classifier.
        weight_decay:  L2 regularization for Adam optimizer.
        pretrained_lr_ratio: LR multiplier for pretrained backbone params (e.g. 0.3).
        unfreeze_after_epoch: Unfreeze last ResNet stage after this many epochs.
    """
    resume_path = os.path.join(ckpt_dir, f"resume_{name}.pth")
    os.makedirs(ckpt_dir, exist_ok=True)
    gen_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)
    cls_criterion = nn.CrossEntropyLoss() if cls_weight > 0 else None

    # ── Differential Learning Rates ────────────────────────────────
    pretrained_params = []
    scratch_params = []
    has_pretrained = hasattr(model, 'image_encoder') and getattr(model.image_encoder, 'pretrained', False)

    if has_pretrained:
        # At init, ResNet is fully frozen → get_pretrained_params() returns [].
        # We still separate proj (scratch_encoder) from the rest (scratch_other).
        # All trainable params of image_encoder (proj) + rest of model.
        encoder_scratch = list(model.image_encoder.get_scratch_params())
        encoder_scratch_ids = {id(p) for p in encoder_scratch}
        other_params = [p for p in model.parameters() 
                        if p.requires_grad and id(p) not in encoder_scratch_ids]

        backbone_lr = lr * pretrained_lr_ratio
        logger.info(f"[{name}] Pretrained CNN — Backbone frozen, will unfreeze at epoch {unfreeze_after_epoch}")
        logger.info(f"[{name}] Scratch LR: {lr:.1e}, WD: {weight_decay}")
        
        # Single param group for all trainable params (backbone is frozen)
        all_trainable = encoder_scratch + other_params
        optimizer = optim.Adam(all_trainable, lr=lr, weight_decay=weight_decay)
        optimizer.param_groups[0]['_target_lr'] = lr
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.param_groups[0]['_target_lr'] = lr

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=patience)

    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and is_cuda))

    best_f1: float = 0.0
    start_epoch: int = 1
    last_metrics: dict[str, float] = {k: 0.0 for k in [
        "accuracy", "em", "f1", "meteor", "bleu4", "wups",
    ]}
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "val_acc": [], "val_em": [], "val_f1": [], "val_meteor": [],
        "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
        "val_wups": [], "val_top1": [], "val_top5": [],
    }

    # ── Resume from crash checkpoint if available ──────────────────
    if os.path.exists(resume_path):
        print(f"  ⏩ Resuming {name} from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_f1 = ckpt.get("best_f1", 0.0)
        start_epoch = ckpt["epoch"] + 1  # resume from NEXT epoch
        history = ckpt.get("history", history)
        stopper.best_score = ckpt.get("stopper_best", None)
        stopper.counter = ckpt.get("stopper_counter", 0)
        print(f"  ⏩ Resumed at epoch {start_epoch}, best_f1={best_f1:.4f}")
        del ckpt
        if is_cuda:
            torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs + 1):
        # ── Warmup ─────────────────────────────────────────────────
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = g.get('_target_lr', lr) * warmup_factor

        # Linear TF decay (stable for small datasets)
        tf = max(tf_end, tf_start - (tf_start - tf_end) * (epoch - 1) / max(epochs - 1, 1))

        # ── TRAIN ──────────────────────────────────────────────────
        model.train()
        
        # ── Gradual unfreeze for pretrained models ─────────────────
        # (done AFTER model.train() so BN requires_grad is already set correctly)
        if has_pretrained and epoch == unfreeze_after_epoch:
            model.image_encoder.unfreeze_backbone(num_layers=1)
            # model.train() already ran, so BN weight/bias are frozen again.
            # Collect only the conv/other params that are truly trainable.
            existing_ids = set()
            for g in optimizer.param_groups:
                for p in g['params']:
                    existing_ids.add(id(p))
            new_params = [p for p in model.image_encoder.cnn.parameters() 
                          if p.requires_grad and id(p) not in existing_ids]
            if new_params:
                backbone_lr = lr * pretrained_lr_ratio
                optimizer.add_param_group({
                    'params': new_params, 
                    'lr': backbone_lr,
                    '_target_lr': backbone_lr,
                    'weight_decay': weight_decay,
                })
                # Register initial_lr so CosineAnnealingLR doesn't crash
                optimizer.param_groups[-1]['initial_lr'] = backbone_lr
                logger.info(f"  Added {len(new_params)} unfrozen backbone params (lr={backbone_lr:.1e})")

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}",
                    leave=False, mininterval=1.0, ncols=80)

        for batch_idx, (imgs, qs, ql, ans, al, ans_txt, raw_qs) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            qs   = qs.to(device, non_blocking=True)
            ql   = ql.to(device, non_blocking=True)
            ans  = ans.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast('cuda', enabled=(use_amp and is_cuda)):
                out, cls_logits = model(imgs, qs, ql, ans, tf_ratio=tf, raw_questions=raw_qs)
                loss = gen_criterion(out.reshape(-1, out.size(-1)), ans[:, 1:].reshape(-1))

                # Multi-task classification loss
                if cls_weight > 0 and cls_logits is not None and answer_to_idx is not None:
                    from src.data.preprocessing import majority_answer
                    cls_targets = []
                    for ref in ans_txt:
                        maj = majority_answer(ref) if isinstance(ref, list) else ref
                        idx = answer_to_idx.get(maj, -1)
                        cls_targets.append(idx)
                    cls_targets = torch.tensor(cls_targets, device=device)
                    valid_mask = cls_targets >= 0
                    if valid_mask.any():
                        loss = loss + cls_weight * cls_criterion(
                            cls_logits[valid_mask], cls_targets[valid_mask],
                        )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if batch_idx % 10 == 0:  # update progress every 10 batches
                pbar.set_postfix_str(f"loss={loss.item():.3f}")

        train_loss = running_loss / len(train_loader)

        # ── VALIDATE ───────────────────────────────────────────────
        do_full_eval = (epoch % eval_every == 0) or (epoch == epochs)
        model.eval()
        val_loss_sum = 0.0
        preds_all, refs_all = [], []
        all_cls_logits, all_cls_targets = [], []

        with torch.no_grad():
            for imgs, qs, ql, ans, al, ans_txt, raw_qs in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                qs   = qs.to(device, non_blocking=True)
                ql   = ql.to(device, non_blocking=True)
                ans  = ans.to(device, non_blocking=True)

                with torch.autocast('cuda', enabled=(use_amp and is_cuda)):
                    # Use tf_ratio=1.0 for val loss — teacher-forced loss is stable
                    # and comparable to train loss. Free-running (tf=0) causes
                    # error accumulation that produces misleading val loss values.
                    out, cls_logits = model(imgs, qs, ql, ans, tf_ratio=1.0, raw_questions=raw_qs)
                    val_loss_sum += gen_criterion(
                        out.reshape(-1, out.size(-1)), ans[:, 1:].reshape(-1),
                    ).item()

                if do_full_eval:
                    gen = model.generate(imgs, qs, ql, use_beam=use_beam, beam_width=beam_w, raw_questions=raw_qs)
                    for i in range(gen.size(0)):
                        preds_all.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                        refs_all.append(ans_txt[i])

                    # Collect classification logits for Top-K
                    if cls_logits is not None and answer_to_idx is not None:
                        from src.data.preprocessing import majority_answer
                        all_cls_logits.append(cls_logits.cpu())
                        for ref in ans_txt:
                            maj = majority_answer(ref) if isinstance(ref, list) else ref
                            all_cls_targets.append(answer_to_idx.get(maj, -1))

        val_loss = val_loss_sum / len(val_loader)

        if do_full_eval:
            m = batch_metrics(preds_all, refs_all)
            # Top-K accuracy from classifier
            if all_cls_logits:
                cat_logits = torch.cat(all_cls_logits, 0)
                cat_targets = torch.tensor(all_cls_targets)
                valid = cat_targets >= 0
                if valid.any():
                    m["top1"] = compute_topk_accuracy(cat_logits[valid], cat_targets[valid], k=1)
                    m["top5"] = compute_topk_accuracy(cat_logits[valid], cat_targets[valid], k=5)
                else:
                    m["top1"] = m["top5"] = 0.0
            else:
                m["top1"] = m["top5"] = 0.0
            last_metrics = m
        else:
            m = last_metrics

        if epoch > warmup_epochs:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)
        for k in [
            "val_acc", "val_em", "val_f1", "val_meteor",
            "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4",
            "val_wups", "val_top1", "val_top5",
        ]:
            metric_key = k.replace("val_", "").replace("acc", "accuracy")
            history[k].append(m.get(metric_key, 0.0))

        # Compact progress line — visible in Kaggle output
        print(f"  [{name}] Ep {epoch:>2d}/{epochs} "
              f"loss={train_loss:.3f}/{val_loss:.3f} "
              f"F1={m['f1']:.3f} B4={m['bleu4']:.3f} lr={cur_lr:.1e}")

        if do_full_eval and m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_f1": best_f1},
                os.path.join(ckpt_dir, f"best_{name}.pth"),
            )
            print(f"    ★ Saved best (F1={best_f1:.4f})")

        # ── Save resume checkpoint every epoch (crash-safe) ────────
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "history": history,
            "stopper_best": stopper.best_score,
            "stopper_counter": stopper.counter,
        }, resume_path)

        if do_full_eval and stopper(m["f1"]):
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Clean up resume checkpoint after successful training ───────
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print(f"  ✓ Removed resume checkpoint (training complete)")

    return history