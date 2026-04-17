#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any


_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer is)\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:最终答案|答案)\s*[:：]\s*(.+)$", re.IGNORECASE),
]


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _normalize(text: Any) -> str:
    text = _to_text(text)
    text = text.replace("\u3000", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \n\t\r.,。，:：;；!！?？'\"`“”‘’()[]{}")
    return text


def _append_candidate(candidates: list[str], value: Any) -> None:
    normalized = _normalize(value)
    if normalized and normalized not in candidates:
        candidates.append(normalized)


def _extract_candidates(text: Any) -> list[str]:
    raw_text = _to_text(text)
    normalized = _normalize(raw_text)
    if not normalized:
        return []

    candidates: list[str] = []
    _append_candidate(candidates, normalized)

    boxed_matches = _BOXED_PATTERN.findall(raw_text)
    if boxed_matches:
        _append_candidate(candidates, boxed_matches[-1])

    answer_tag_matches = _ANSWER_TAG_PATTERN.findall(raw_text)
    if answer_tag_matches:
        _append_candidate(candidates, answer_tag_matches[-1])

    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            _append_candidate(candidates, match.group(1))

    non_empty_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if non_empty_lines:
        _append_candidate(candidates, non_empty_lines[-1])

    return candidates


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred_candidates = _extract_candidates(solution_str)
    gt_candidates = _extract_candidates(ground_truth)

    if not gt_candidates or not pred_candidates:
        return 0.0

    if set(pred_candidates) & set(gt_candidates):
        return 1.0

    return 0.0