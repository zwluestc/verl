#!/usr/bin/env python3
from __future__ import annotations

import re
import os
import json
import requests
from typing import Any

# ================= DeepSeek LLM Judge Settings =================
JUDGE_SYSTEM_PROMPT = """You are a strict but fair mathematics and physics professor.
Your task is to judge whether a student's final answer is mathematically equivalent to the reference final answer.

Rules:
1. Compare ONLY the final mathematical conclusions, not the derivation paths.
2. Different notations, equivalent formulas, or different order of terms are acceptable if mathematically equivalent.
3. If the candidate misses critical terms, has wrong signs, wrong constants, or reaches an incorrect final formula, mark as WRONG.
4. If the candidate gives no recognizable final answer, mark as WRONG.
5. Output MUST be valid JSON: {"correct": true/false, "reason": "brief explanation"}
6. Do NOT output thinking tags like <think>. Output JSON directly.
"""

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4"
_FINAL_TAG_PATTERN = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)


def llm_judge(question: str, reference: str, candidate: str) -> float:
    prompt = f"""[Question]
{question}

[Reference Final Answer]
{reference}

[Student Final Answer]
{candidate}

Judge: Does the Student Final Answer reach a conclusion equivalent to the Reference Final Answer?
Think step by step, then respond with JSON only: {{"correct": true/false, "reason": "..."}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    try:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
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
        print(f"[DeepSeek LLM Judge Warning] Request failed or JSON parsing error: {e}.")
        return -1.0
# =======================================================


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
    text = re.sub(r"\s+", " ", text).strip(" \n\t\r.,。，:：;；!！?？'\"`\"""''()[]").replace(" ", "")
    return text


def _extract_last_nonempty_final(text: Any) -> str:
    matches = _FINAL_TAG_PATTERN.findall(_to_text(text))
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate:
            return candidate
    return ""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred_text = _to_text(solution_str)
    gt_text = _to_text(ground_truth)

    # Enforce training target format: candidate must provide a non-empty <final>...</final>.
    pred_final = _extract_last_nonempty_final(pred_text)
    if not pred_final:
        return 0.0

    # The GRPO dataset should already store the extracted <final> content as ground_truth.
    # If a tagged string is passed in, still normalize to the final block for safety.
    gt_final = _extract_last_nonempty_final(gt_text) or gt_text.strip()
    if not gt_final:
        return 0.0

    # 1. Exact-match fast path on normalized final answers only.
    if _normalize(pred_final) == _normalize(gt_final):
        return 1.0

    # 2. LLM-as-judge on final answers only.
    question = kwargs.get("prompt_str", "Unknown Question")
    if question == "Unknown Question" and isinstance(extra_info, dict):
        question = extra_info.get("prompt", "Unknown Question")

    llm_score = llm_judge(question, gt_final, pred_final)
    if llm_score != -1.0:
        return llm_score

    # 3. Conservative fallback: binary exact match only.
    return 1.0 if _normalize(pred_final) == _normalize(gt_final) else 0.0
