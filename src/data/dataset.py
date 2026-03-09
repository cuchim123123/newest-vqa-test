"""
A-OKVQA Dataset, Vocabulary, and DataLoader utilities.

Provides a Vocabulary class with stoi/itos interface (compatible with
vqa_s1 decode_sequence), an A-OKVQA PyTorch Dataset that loads images
on-the-fly from HuggingFace, and a collate function for variable-length
sequences.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.preprocessing import (
    normalize_answer, tokenize, extract_answer,
    extract_all_references,
)

# ════════════════════════════════════════════════════════════════════
# Special token indices (module-level constants for fast import)
# ════════════════════════════════════════════════════════════════════
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# ════════════════════════════════════════════════════════════════════
# Vocabulary
# ════════════════════════════════════════════════════════════════════
class Vocabulary:
    """Bidirectional word ↔ index mapping with special tokens.

    Exposes both ``stoi`` / ``itos`` (dict-style) attributes for
    compatibility with the rest of the codebase (decode_sequence uses
    ``vocab.itos``).
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

    def __init__(self, freq_threshold: int = 3) -> None:
        self.freq_threshold = freq_threshold
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._counter: Counter = Counter()

        for token in self.SPECIAL_TOKENS:
            self._add_token(token)

    # ── internal helpers ──────────────────────────────────────────
    def _add_token(self, token: str) -> int:
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token
            return idx
        return self.stoi[token]

    # ── public API ────────────────────────────────────────────────
    def build_vocabulary(self, sentences: List[str]) -> None:
        """Count word frequencies from a list of sentences, then add
        words that meet ``freq_threshold`` to the vocabulary."""
        for sent in sentences:
            for tok in tokenize(sent):
                self._counter[tok] += 1
        for word, freq in self._counter.items():
            if freq >= self.freq_threshold:
                self._add_token(word)

    def numericalize(self, text: str) -> List[int]:
        """Convert a raw string to a list of token indices."""
        return [self.stoi.get(tok, UNK_IDX) for tok in tokenize(text)]

    def encode_with_special(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text with <SOS> ... <EOS> framing and optional padding."""
        indices = [SOS_IDX] + self.numericalize(text) + [EOS_IDX]
        if max_len is not None:
            indices = indices[:max_len]
            indices += [PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.stoi)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, threshold={self.freq_threshold})"


# ════════════════════════════════════════════════════════════════════
# A-OKVQA Dataset
# ════════════════════════════════════════════════════════════════════
class AOKVQA_Dataset(Dataset):
    """PyTorch Dataset for A-OKVQA.

    Each item yields:
        (image_tensor, question_indices, question_length,
         answer_indices, answer_length, reference_texts, raw_question_str)
    """

    def __init__(
        self,
        data: List[dict],
        question_vocab: Vocabulary,
        answer_vocab: Vocabulary,
        transform=None,
    ) -> None:
        self.data = data
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        # ── Image ─────────────────────────────────────────────────
        image = item.get("image")
        if image is None:
            # Try to load from image bytes (cached)
            image = Image.new("RGB", (224, 224))
        elif not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # ── Question ──────────────────────────────────────────────
        raw_question = item.get("question", "")
        q_tokens = [SOS_IDX] + self.question_vocab.numericalize(raw_question) + [EOS_IDX]
        q_tensor = torch.tensor(q_tokens, dtype=torch.long)
        q_length = len(q_tokens)

        # ── Answer (target for decoder) ───────────────────────────
        # Use expanded rationale target if available (from expand_data_with_rationales)
        target_answer = item.get("_target_answer") or extract_answer(item)
        a_tokens = [SOS_IDX] + self.answer_vocab.numericalize(target_answer) + [EOS_IDX]
        a_tensor = torch.tensor(a_tokens, dtype=torch.long)
        a_length = len(a_tokens)

        # ── Reference answers (for evaluation metrics) ────────────
        refs = extract_all_references(item)

        return image, q_tensor, q_length, a_tensor, a_length, refs, raw_question


# ════════════════════════════════════════════════════════════════════
# Collate function (variable-length padding)
# ════════════════════════════════════════════════════════════════════
def collate_fn(batch):
    """Pad variable-length sequences in a batch.

    Returns:
        images:      (B, C, H, W)
        questions:   (B, max_q_len) — padded
        q_lengths:   (B,)
        answers:     (B, max_a_len) — padded
        a_lengths:   (B,)
        refs:        list[list[str]] — raw reference answer strings
        raw_qs:      list[str] — raw question strings
    """
    images, q_tensors, q_lengths, a_tensors, a_lengths, refs, raw_qs = zip(*batch)

    images = torch.stack(images, 0)
    q_lengths = torch.tensor(q_lengths, dtype=torch.long)
    a_lengths = torch.tensor(a_lengths, dtype=torch.long)

    questions = pad_sequence(q_tensors, batch_first=True, padding_value=PAD_IDX)
    answers = pad_sequence(a_tensors, batch_first=True, padding_value=PAD_IDX)

    return images, questions, q_lengths, answers, a_lengths, list(refs), list(raw_qs)
