"""
Text preprocessing utilities for VQA.

Provides cleaning, tokenisation, answer normalisation, question-type
classification, and A-OKVQA–specific data expansion helpers.
Best-of-both from vqa_project + vqa_s1.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, List


# ════════════════════════════════════════════════════════════════════
# Core text processing
# ════════════════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """Normalise an answer string for metric computation.

    Lowercase, strip whitespace and punctuation, collapse spaces.
    Compatible with VQA evaluation standards.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation except apostrophe
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokeniser on normalised text."""
    return normalize_answer(text).split()


# ════════════════════════════════════════════════════════════════════
# Answer extraction helpers (A-OKVQA specific)
# ════════════════════════════════════════════════════════════════════

def majority_answer(refs) -> str:
    """Return the most frequent answer from a list of references."""
    if isinstance(refs, str):
        return normalize_answer(refs)
    if not refs:
        return ""
    normalised = [normalize_answer(r) for r in refs if r]
    if not normalised:
        return ""
    counter = Counter(normalised)
    return counter.most_common(1)[0][0]


def extract_answer(item: dict) -> str:
    """Extract the best answer string from an A-OKVQA item.

    Priority: rationales > direct_answers > choices[correct_choice_idx].
    """
    # Prefer rationales for generative training (longer, more natural)
    rationales = item.get("rationales", [])
    if rationales and rationales[0]:
        return rationales[0]

    # Fall back to direct answers (list of annotator answers)
    direct = item.get("direct_answers", [])
    if direct:
        return majority_answer(direct)

    # Fall back to multiple choice
    choices = item.get("choices", [])
    correct_idx = item.get("correct_choice_idx")
    if choices and correct_idx is not None and correct_idx < len(choices):
        return choices[correct_idx]

    return ""


def extract_all_references(item: dict) -> List[str]:
    """Extract ALL valid reference answers for evaluation (multi-ref)."""
    refs = []

    # Direct answers from annotators
    direct = item.get("direct_answers", [])
    if direct:
        refs.extend([a for a in direct if a])

    # Rationales
    rationales = item.get("rationales", [])
    if rationales:
        refs.extend([r for r in rationales if r])

    # Multiple choice correct answer
    choices = item.get("choices", [])
    correct_idx = item.get("correct_choice_idx")
    if choices and correct_idx is not None and correct_idx < len(choices):
        refs.append(choices[correct_idx])

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for r in refs:
        norm = normalize_answer(r)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(r)

    return unique if unique else [""]


# ════════════════════════════════════════════════════════════════════
# Data expansion with rationales (3× augmentation)
# ════════════════════════════════════════════════════════════════════

def expand_data_with_rationales(data: List[dict]) -> List[dict]:
    """Expand training data by creating copies with each rationale as answer.

    A-OKVQA items often have 3 rationale annotations. This creates up to 3×
    training samples, teaching the decoder to generate longer, more natural
    explanatory answers.
    """
    expanded = []
    for item in data:
        rationales = item.get("rationales", [])
        if rationales:
            for rat in rationales:
                if rat and rat.strip():
                    new_item = dict(item)
                    new_item["_target_answer"] = rat
                    expanded.append(new_item)
        else:
            item["_target_answer"] = extract_answer(item)
            expanded.append(item)
    return expanded


# ════════════════════════════════════════════════════════════════════
# Question type classification
# ════════════════════════════════════════════════════════════════════

_QUESTION_TYPE_PATTERNS = [
    ("yes/no", re.compile(r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b", re.IGNORECASE)),
    ("how many", re.compile(r"^how many\b", re.IGNORECASE)),
    ("how", re.compile(r"^how\b", re.IGNORECASE)),
    ("what color", re.compile(r"^what (?:color|colour)\b", re.IGNORECASE)),
    ("what kind", re.compile(r"^what (?:kind|type|sort)\b", re.IGNORECASE)),
    ("what", re.compile(r"^what\b", re.IGNORECASE)),
    ("where", re.compile(r"^where\b", re.IGNORECASE)),
    ("when", re.compile(r"^when\b", re.IGNORECASE)),
    ("who", re.compile(r"^who\b", re.IGNORECASE)),
    ("which", re.compile(r"^which\b", re.IGNORECASE)),
    ("why", re.compile(r"^why\b", re.IGNORECASE)),
]


def classify_question(question: str) -> str:
    """Classify a question into a high-level type for analysis."""
    q = question.strip()
    for label, pattern in _QUESTION_TYPE_PATTERNS:
        if pattern.search(q):
            return label
    return "other"
