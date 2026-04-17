#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _normalize(text: Any) -> str:
    text = _to_text(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _extract_boxed(text: str) -> str:
    matched = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return _normalize(matched[-1]) if matched else ""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred = _normalize(solution_str)
    gt = _normalize(ground_truth)

    if not gt:
        return 0.0
    if pred == gt:
        return 1.0

    pred_boxed = _extract_boxed(pred)
    gt_boxed = _extract_boxed(gt)
    if pred_boxed and gt_boxed and pred_boxed == gt_boxed:
        return 1.0
    if pred_boxed and pred_boxed == gt:
        return 1.0
    return 0.0