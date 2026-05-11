#!/usr/bin/env python3
from __future__ import annotations

import re
import os
import json
import requests
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

# ================= LLM Judge Settings =================
JUDGE_SYSTEM_PROMPT = """You are a strict but fair mathematics and physics professor.
Your task is to judge whether a student's reasoning and final answer are mathematically equivalent to the reference answer.

Rules:
1. The reference contains a full derivation and a final answer. The final answer is the ground truth.
2. The candidate also contains a full derivation and a final answer.
3. Judge whether the candidate's FINAL CONCLUSION is correct and equivalent to the reference's final conclusion.
4. The student may use different notations, different derivation paths, or present steps in different order; these are acceptable if mathematically equivalent.
5. If the candidate is partially correct but misses critical terms, has wrong signs, wrong constants, or reaches an incorrect final formula, mark as WRONG.
6. If the candidate gives no recognizable final answer, or the derivation is completely off-track, mark as WRONG.
7. Output MUST be valid JSON: {"correct": true/false, "reason": "brief explanation"}
8. Do NOT output thinking tags like <think>. Output JSON directly.
"""

def llm_judge(question: str, reference: str, candidate: str) -> float:
    prompt = f"""[Question]
{question}

[Reference Answer]
{reference}

[Student Answer]
{candidate}

Judge: Does the Student Answer reach a conclusion equivalent to the Reference Answer?
Think step by step, then respond with JSON only: {{"correct": true/false, "reason": "..."}}
"""
    api_url = os.environ.get("JUDGE_API_URL", "http://localhost:8000/v1/chat/completions")
    # For a local vLLM, model name doesn't matter much if it's the only one loaded
    model_name = os.environ.get("JUDGE_MODEL", "qwen3")
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        raw_text = result['choices'][0]['message']['content'].strip()
        
        # Remove <think> and code blocks recursively
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        if cleaned and cleaned[0] != "{":
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end+1]

        data = json.loads(cleaned)
        return 1.0 if data.get("correct", False) else 0.0
    except Exception as e:
        print(f"[LLM Judge Warning] Request failed or JSON parsing error: {e}. Falling back to Rule-Based EM.")
        return -1.0
# =======================================================


_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer is)\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:最终答案|答案)\s*[:：]\s*(.+)$", re.IGNORECASE),
]
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+(?:\.\d+)?")

def _to_text(value: Any) -> str:
    return "" if value is None else (value if isinstance(value, str) else str(value))

def _normalize(text: Any) -> str:
    text = _to_text(text).replace("\u3000", " ").replace("\xa0", " ").lower()
    text = text.replace("\\dfrac", "\\frac").replace("\\left(", "(").replace("\\right)", ")")
    text = text.replace("\\left[", "[").replace("\\right]", "]")
    text = text.replace("\\left\\{", "{").replace("\\right\\}", "}")
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("^{2}", "^2").replace("^{3}", "^3")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    text = text.replace("\\ ", "").replace("{", "").replace("}", "") 
    text = re.sub(r"\s+", " ", text).strip(" \n\t\r.,。，:：;；!！?？'\"`“”‘’()[]").replace(" ", "")
    return text

def extract_boxed_content(text: str) -> list[str]:
    candidates, start_idx = [], 0
    while True:
        start_idx = text.find(r"\boxed", start_idx)
        if start_idx == -1: break
        brace_start = text.find('{', start_idx)
        if brace_start == -1 or brace_start > start_idx + 8:
            start_idx += 6
            continue
        brace_count, end_idx = 0, -1
        for i in range(brace_start, len(text)):
            if text[i] == '{': brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i; break
        if end_idx != -1: candidates.append(text[brace_start+1:end_idx].strip())
        start_idx = brace_start + 1
    return candidates

def _append_candidate(candidates: list[str], value: Any) -> None:
    normalized = _normalize(value)
    if normalized and normalized not in candidates: candidates.append(normalized)

def _extract_candidates(text: Any) -> list[str]:
    raw_text = _to_text(text)
    normalized = _normalize(raw_text)
    if not normalized: return []
    candidates: list[str] = []
    _append_candidate(candidates, normalized)
    for match in extract_boxed_content(raw_text):
        _append_candidate(candidates, match)
    answer_tag_matches = _ANSWER_TAG_PATTERN.findall(raw_text)
    if answer_tag_matches: _append_candidate(candidates, answer_tag_matches[-1])
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(raw_text)
        if match: _append_candidate(candidates, match.group(1))
    non_empty_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if non_empty_lines:
        _append_candidate(candidates, non_empty_lines[-1])
        _append_candidate(candidates, " ".join(non_empty_lines[-3:]))
    paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n", raw_text) if seg.strip()]
    if paragraphs: _append_candidate(candidates, paragraphs[-1])
    return candidates

def _tokenize(text: Any) -> list[str]:
    return _TOKEN_PATTERN.findall(_normalize(text))

def _content_tokens(text: Any) -> list[str]:
    return [tok for tok in _tokenize(text) if tok.isdigit() or len(tok) > 1 or re.match(r"[\u4e00-\u9fff]", tok)]

def _token_f1(pred: Any, gt: Any) -> float:
    pred_tokens, gt_tokens = _tokenize(pred), _tokenize(gt)
    if not pred_tokens or not gt_tokens: return 0.0
    pred_counter, gt_counter = Counter(pred_tokens), Counter(gt_tokens)
    overlap = sum((pred_counter & gt_counter).values())
    if overlap == 0: return 0.0
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gt_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)

def _string_similarity(pred: Any, gt: Any) -> float:
    pn, gn = _normalize(pred), _normalize(gt)
    return 0.0 if not pn or not gn else SequenceMatcher(None, pn, gn).ratio()

def _keyword_recall(pred: Any, gt: Any) -> float:
    gt_counter = Counter(_content_tokens(gt))
    if not gt_counter: return 0.0
    overlap = sum((Counter(_content_tokens(pred)) & gt_counter).values())
    return overlap / max(sum(gt_counter.values()), 1)

def _best_candidate_score(pred_candidates: list[str], gt_candidates: list[str]) -> float:
    best = 0.0
    for pred in pred_candidates:
        for gt in gt_candidates:
            if pred == gt: return 1.0
            best = max(best, 0.6 * _token_f1(pred, gt) + 0.2 * _string_similarity(pred, gt) + 0.2 * _keyword_recall(pred, gt))
    return best

def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred_text = _to_text(solution_str)
    gt_text = _to_text(ground_truth)
    
    pred_candidates = _extract_candidates(pred_text)
    gt_candidates = _extract_candidates(gt_text)

    # 1. Rule-based exact match Fast Path
    if pred_candidates and gt_candidates:
        if _normalize(pred_text) == _normalize(gt_text):
            return 1.0
        candidate_score = _best_candidate_score(pred_candidates, gt_candidates)
        if candidate_score >= 0.98:
            return 1.0
            
    # 2. LLM as Judge
    question = kwargs.get("prompt_str", "Unknown Question")
    # if prompt_str not provided directly, try to get from extra_info if injected there
    if question == "Unknown Question" and isinstance(extra_info, dict):
         question = extra_info.get("prompt", "Unknown Question")
         
    llm_score = llm_judge(question, gt_text, pred_text)
    if llm_score != -1.0:
        return llm_score

    # 3. Fallback Rule-based if LLM fails
    if not pred_candidates or not gt_candidates:
        return 0.0
    candidate_score = _best_candidate_score(pred_candidates, gt_candidates)
    full_text_score = 0.5 * _token_f1(pred_text, gt_text) + 0.2 * _string_similarity(pred_text, gt_text) + 0.3 * _keyword_recall(pred_text, gt_text)
    
    if len(_tokenize(pred_text)) < 24:
        return round(min(0.5 * candidate_score + 0.5 * full_text_score, 0.35), 4)

    final_score = 0.7 * candidate_score + 0.3 * full_text_score
    return round(min(max(final_score, 0.0), 1.0), 4)
