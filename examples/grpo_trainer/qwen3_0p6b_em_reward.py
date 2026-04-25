#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer is)\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:最终答案|答案)\s*[:：]\s*(.+)$", re.IGNORECASE),
]
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+(?:\.\d+)?")

def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)

def _normalize(text: Any) -> str:
    """强化版文本清理：增强了对数学公式格式等价的处理"""
    text = _to_text(text)
    # 基础字符清理
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.lower()
    
    # LaTeX 数学格式归一化（降低由于书写习惯不同造成的等价但序列不同的影响）
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\left(", "(").replace("\\right)", ")")
    text = text.replace("\\left[", "[").replace("\\right]", "]")
    text = text.replace("\\left\\{", "{").replace("\\right\\}", "}")
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("^{2}", "^2").replace("^{3}", "^3")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    text = text.replace("\\ ", "")
    
    # 粗暴去除大括号，避免只因为 \text{a} 和 \text a 这种细微差异导致匹配失败
    text = text.replace("{", "").replace("}", "") 
    
    # 统一连续空白，并去除两端噪音
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \n\t\r.,。，:：;；!！?？'\"`“”‘’()[]")
    
    # 去除内部所有空格，提升纯公式直接对比的命中率
    text = text.replace(" ", "")
    return text

def extract_boxed_content(text: str) -> list[str]:
    """提取带有嵌套大括号的 \\boxed{} 内容 (修复了正则无法提取嵌套的问题)"""
    candidates = []
    start_idx = 0
    while True:
        start_idx = text.find(r"\boxed", start_idx)
        if start_idx == -1:
            break
        
        # 寻找紧接 \boxed 之后的左大括号
        brace_start = text.find('{', start_idx)
        if brace_start == -1 or brace_start > start_idx + 8:
            start_idx += 6
            continue
            
        brace_count = 0
        end_idx = -1
        # 基于堆栈思想进行准确的括号匹配
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            candidates.append(text[brace_start+1:end_idx].strip())
        start_idx = brace_start + 1
    return candidates

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

    boxed_matches = extract_boxed_content(raw_text)
    for match in boxed_matches:
        _append_candidate(candidates, match)

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
        _append_candidate(candidates, " ".join(non_empty_lines[-3:]))

    paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n", raw_text) if seg.strip()]
    if paragraphs:
        _append_candidate(candidates, paragraphs[-1])

    return candidates

def _tokenize(text: Any) -> list[str]:
    normalized = _normalize(text)
    return _TOKEN_PATTERN.findall(normalized)

def _content_tokens(text: Any) -> list[str]:
    tokens = _tokenize(text)
    return [tok for tok in tokens if tok.isdigit() or len(tok) > 1 or re.match(r"[\u4e00-\u9fff]", tok)]

def _token_f1(pred: Any, gt: Any) -> float:
    pred_tokens = _tokenize(pred)
    gt_tokens = _tokenize(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    overlap = sum((pred_counter & gt_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gt_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)

def _string_similarity(pred: Any, gt: Any) -> float:
    pred_norm = _normalize(pred)
    gt_norm = _normalize(gt)
    if not pred_norm or not gt_norm:
        return 0.0
    return SequenceMatcher(None, pred_norm, gt_norm).ratio()

def _keyword_recall(pred: Any, gt: Any) -> float:
    pred_counter = Counter(_content_tokens(pred))
    gt_counter = Counter(_content_tokens(gt))
    if not gt_counter:
        return 0.0
    overlap = sum((pred_counter & gt_counter).values())
    total = sum(gt_counter.values())
    return overlap / max(total, 1)

def _best_candidate_score(pred_candidates: list[str], gt_candidates: list[str]) -> float:
    best = 0.0
    for pred in pred_candidates:
        for gt in gt_candidates:
            if pred == gt:
                return 1.0
            best = max(
                best,
                0.6 * _token_f1(pred, gt) + 0.2 * _string_similarity(pred, gt) + 0.2 * _keyword_recall(pred, gt),
            )
    return best

def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred_text = _to_text(solution_str)
    gt_text = _to_text(ground_truth)
    
    pred_candidates = _extract_candidates(pred_text)
    gt_candidates = _extract_candidates(gt_text)

    if not pred_candidates or not gt_candidates:
        return 0.0

    pred_norm = _normalize(pred_text)
    gt_norm = _normalize(gt_text)
    if pred_norm == gt_norm:
        return 1.0

    candidate_score = _best_candidate_score(pred_candidates, gt_candidates)
    
    # 如果核心候选答案匹配极高，说明已经得到了正确答案，给予满分（不再因为胡言论语或其他内容倒扣）
    if candidate_score >= 0.98:
        return 1.0

    full_f1 = _token_f1(pred_text, gt_text)
    full_similarity = _string_similarity(pred_text, gt_text)
    keyword_recall = _keyword_recall(pred_text, gt_text)

    full_text_score = 0.5 * full_f1 + 0.2 * full_similarity + 0.3 * keyword_recall

    # 针对极短输出且候选匹配得分低进行惩罚（防止随便生成几个字符“蹭” F1 分数）
    if len(_tokenize(pred_text)) < 24:
        short_score = 0.5 * candidate_score + 0.5 * full_text_score
        return round(min(short_score, 0.35), 4)

    # 去废除了以前阻碍长过程验证的 _length_ratio 倒扣惩罚
    # 更加关注于候选答案质量(权重 0.7)，与推导过程与标准答案全文本的语义覆盖率(权重 0.3)
    final_score = 0.7 * candidate_score + 0.3 * full_text_score
    
    return round(min(max(final_score, 0.0), 1.0), 4)
