"""Visualization utilities updated for Dual Attention (Spatial + Text)."""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX
from src.data.preprocessing import normalize_answer, majority_answer, classify_question

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
logger = logging.getLogger("VQA")

# ═══════════════════════════════════════════════════════════════════════
# 1. Training curves
# ═══════════════════════════════════════════════════════════════════════
def plot_training_curves(all_histories: dict[str, dict[str, list[float]]], save_prefix: str = "fig") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    titles = ["Training Loss", "Validation Loss", "Learning Rate"]
    keys = ["train_loss", "val_loss", "lr"]
    
    for ax, key, title in zip(axes, keys, titles):
        for i, (name, h) in enumerate(all_histories.items()):
            ax.plot(h[key], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}1_loss_lr.png", dpi=200)
    plt.show()

# ═══════════════════════════════════════════════════════════════════════
# 2. Dual Attention Visualization
# ═══════════════════════════════════════════════════════════════════════
def visualize_attention(model, loader, answer_vocab, question_vocab, device, n=3, save_path="fig6_attn.png") -> None:
    if not model.use_attention: return
    model.eval()
    batch = next(iter(loader))
    # collate_fn returns: images, questions, q_lengths, answers, a_lengths, answer_texts, raw_qs
    imgs, qs, ql, ans, al, ans_txt, raw_qs = batch
    imgs_d, qs_d, ql_d = imgs.to(device), qs.to(device), ql.to(device)
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    for idx in range(min(n, len(ans_txt))):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        img_feat = model.image_encoder(imgs_d[idx:idx+1])
        q_out, (h, c), q_mask = model.question_encoder(qs_d[idx:idx+1], ql_d[idx:idx+1])
        tok = torch.tensor([SOS_IDX], device=device)
        text_attn_list, gen_tokens = [], []

        with torch.no_grad():
            for _ in range(15):
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                text_ctx, t_weights = model.answer_decoder.text_attention(h[-1], q_out, q_mask)
                img_ctx, s_weights = model.answer_decoder.spatial_attention(h[-1], img_feat)
                text_attn_list.append(t_weights.cpu().numpy().flatten())
                inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], 2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                pred = model.answer_decoder.fc(out.squeeze(1))
                tok = pred.argmax(1)
                gen_tokens.append(tok.item())
                if tok.item() == EOS_IDX: break

        q_toks = [question_vocab.itos.get(t, "?") for t in qs[idx].tolist() if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        a_toks = [answer_vocab.itos.get(t, "?") for t in gen_tokens if t != EOS_IDX]
        img_show = inv_norm(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(img_show); axes[0].axis("off")
        axes[0].set_title(f"Q: {' '.join(q_toks)}\nPred: {' '.join(a_toks)}", fontsize=10)
        mat = np.array(text_attn_list[:len(a_toks)])[:, :len(q_toks)]
        im = axes[1].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(len(q_toks))); axes[1].set_xticklabels(q_toks, rotation=45)
        axes[1].set_yticks(range(len(a_toks))); axes[1].set_yticklabels(a_toks)
        plt.colorbar(im, ax=axes[1]); plt.show()

# ═══════════════════════════════════════════════════════════════════════
# 3. Missing CLI Helpers
# ═══════════════════════════════════════════════════════════════════════
def plot_radar_chart(test_results: dict[str, dict[str, float]], save_path: str = "fig4_radar.png") -> None:
    metrics = ["accuracy", "em", "f1", "meteor", "bleu4"]
    labels = ["Acc", "EM", "F1", "METEOR", "B-4"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, m) in enumerate(test_results.items()):
        values = [m.get(k, 0.0) for k in metrics] + [m.get(metrics[0], 0.0)]
        ax.plot(angles, values, "o-", label=name, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels); ax.legend()
    plt.savefig(save_path); plt.show()

def plot_bar_chart(test_results: dict[str, dict[str, float]], save_path: str = "fig5_bar.png") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(test_results))
    ax.bar(x, [m.get("f1", 0.0) for m in test_results.values()], color=COLORS[:len(test_results)])
    ax.set_xticks(x); ax.set_xticklabels(test_results.keys()); ax.set_title("F1 Comparison")
    plt.savefig(save_path); plt.show()

def plot_confusion_matrix(preds, refs, questions=None, save_path="fig8_cm.png", top_k: int = 15):
    """
    Vẽ Confusion Matrix cho Top-K câu trả lời phổ biến nhất.
    - preds: list[str]  — câu trả lời dự đoán
    - refs : list[str | list[str]] — câu trả lời đúng (có thể là list)
    - top_k: số lượng nhãn hiển thị (mặc định 15)
    """
    from collections import Counter
    import numpy as np

    # Chuẩn hóa refs về string đơn
    flat_refs = [r[0] if isinstance(r, (list, tuple)) else r for r in refs]

    # Chọn top_k nhãn xuất hiện nhiều nhất trong ground-truth
    counter = Counter(flat_refs)
    top_labels = [lbl for lbl, _ in counter.most_common(top_k)]
    label_set = set(top_labels)
    OTHER = "<other>"

    # Map dự đoán và truth về top_k + <other>
    def _map(s):
        return s if s in label_set else OTHER

    all_labels = top_labels + [OTHER]
    label_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)

    matrix = np.zeros((n, n), dtype=int)
    for p, r in zip(preds, flat_refs):
        row = label_idx[_map(r)]   # true (trục Y)
        col = label_idx[_map(p)]   # pred (trục X)
        matrix[row, col] += 1

    # Normalize theo từng hàng (recall per class)
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    matrix_norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), max(8, n * 0.6)))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n)); ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — Top {top_k} Answers (row-normalized recall)", fontweight="bold")

    # Ghi số lên ô
    for i in range(n):
        for j in range(n):
            val = matrix_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_question_type_analysis(type_results, save_path="fig9_qtype.png"):
    types = list(type_results.keys())
    f1s = [type_results[t]["f1"] for t in types]
    plt.figure(figsize=(12, 6))
    plt.bar(types, f1s, color=COLORS[1])
    plt.title("F1 Score by Question Type"); plt.xticks(rotation=45); plt.show()

def visualize_attention_overlay(model, loader, answer_vocab, question_vocab, device, n=3, save_path="fig10_spatial_attn.png") -> None:
    """Vẽ Heatmap đè lên ảnh gốc để thể hiện Spatial Attention."""
    if not model.use_attention: 
        logger.warning("Mô hình không sử dụng Attention. Bỏ qua vẽ Spatial Overlay.")
        return
        
    model.eval()
    batch = next(iter(loader))
    # collate_fn returns: images, questions, q_lengths, answers, a_lengths, answer_texts, raw_qs
    imgs, qs, ql, ans, al, ans_txt, raw_qs = batch
    imgs_d, qs_d, ql_d = imgs.to(device), qs.to(device), ql.to(device)
    
    # Định nghĩa hàm giải chuẩn hóa để lấy lại ảnh gốc
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for idx in range(min(n, len(ans_txt))):
        # 1. Trích xuất đặc trưng và tính toán Attention
        img_feat = model.image_encoder(imgs_d[idx:idx+1])
        q_out, (h, c), q_mask = model.question_encoder(qs_d[idx:idx+1], ql_d[idx:idx+1])
        tok = torch.tensor([SOS_IDX], device=device)
        
        gen_tokens = []
        spatial_weights_list = [] # Lưu trọng số không gian cho từng từ sinh ra

        with torch.no_grad():
            for step in range(15): # Sinh tối đa 15 từ
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                text_ctx, _ = model.answer_decoder.text_attention(h[-1], q_out, q_mask)
                
                # Tính trọng số không gian s_weights có kích thước (1, 49)
                img_ctx, s_weights = model.answer_decoder.spatial_attention(h[-1], img_feat)
                spatial_weights_list.append(s_weights.cpu().numpy().flatten())
                
                inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], 2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                pred = model.answer_decoder.fc(out.squeeze(1))
                tok = pred.argmax(1)
                
                if tok.item() == EOS_IDX: 
                    break
                gen_tokens.append(tok.item())

        # 2. Xử lý ảnh gốc để vẽ
        q_toks = [question_vocab.itos.get(t, "?") for t in qs[idx].tolist() if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        a_toks = [answer_vocab.itos.get(t, "?") for t in gen_tokens]
        
        # Chuyển tensor ảnh về numpy array (H, W, C)
        img_original = inv_norm(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()
        # Chuyển ảnh float [0,1] về uint8 [0,255] để dùng với OpenCV
        img_uint8 = np.uint8(255 * img_original)

        # 3. Vẽ biểu đồ: Ảnh gốc + Lưới Heatmap cho từng từ sinh ra
        num_words = len(a_toks)
        # Bố cục: 1 hàng, số cột = 1 (ảnh gốc) + số từ sinh ra
        fig, axes = plt.subplots(1, num_words + 1, figsize=(4 * (num_words + 1), 4))
        
        # Tiêu đề chung cho toàn bộ hình
        fig.suptitle(f"Q: {' '.join(q_toks)}", fontsize=16, fontweight="bold")
        
        # Vẽ ảnh gốc ở cột đầu tiên
        axes[0].imshow(img_original)
        axes[0].axis("off")
        axes[0].set_title("Original Image", fontsize=12)

        # Vẽ Heatmap cho từng từ
        for i, word in enumerate(a_toks):
            ax = axes[i + 1]
            # Lấy vector trọng số (49,), chuyển thành ma trận 7x7
            attn_map = spatial_weights_list[i].reshape(7, 7)
            
            # Phóng to ma trận 7x7 lên 224x224 (bằng kích thước ảnh)
            try:
                import cv2
                attn_map_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
            except ImportError:
                from PIL import Image as _PILImage
                attn_map_resized = np.array(
                    _PILImage.fromarray(attn_map).resize((224, 224), _PILImage.BICUBIC)
                )
            
            # Vẽ ảnh gốc làm nền mờ
            ax.imshow(img_original, alpha=0.5)
            # Phủ Heatmap lên trên
            im = ax.imshow(attn_map_resized, cmap='jet', alpha=0.6)
            
            ax.axis("off")
            ax.set_title(f"Focus for: '{word}'", fontsize=14, color='red')
            
        plt.tight_layout()
        plt.savefig(f"{save_path.split('.png')[0]}_{idx}.png", dpi=200)
        plt.show()