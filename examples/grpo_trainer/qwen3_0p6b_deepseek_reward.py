#!/usr/bin/env python3
from __future__ import annotations

import re
import os
import json
import time
import fcntl
from urllib.parse import urlparse
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


def _first_nonempty_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _build_chat_completions_url() -> str:
    explicit_url = _first_nonempty_env("LLM_API_URL", "DEEPSEEK_API_URL")
    if explicit_url:
        return explicit_url

    base_url = _first_nonempty_env("LLM_API_BASE_URL", "BASE_URL", "DEEPSEEK_API_BASE_URL")
    if not base_url:
        return "https://api.deepseek.com/v1/chat/completions"

    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return base_url
    if path.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


DEEPSEEK_API_KEY = _first_nonempty_env("LLM_API_KEY", "API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = _build_chat_completions_url()
DEEPSEEK_MODEL = _first_nonempty_env("LLM_JUDGE_MODEL", "DEEPSEEK_MODEL") or "deepseek-v4-flash"
_FINAL_TAG_PATTERN = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)
_CORRECT_FLAG_PATTERN = re.compile(r'"correct"\s*:\s*(true|false)', re.IGNORECASE)
REWARD_DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
REWARD_DEBUG_LIMIT = min(int(os.environ.get("REWARD_DEBUG_LIMIT", "1000")), 1000)


def _clean_judge_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _extract_correct_flag(text: str) -> float | None:
    text = _clean_judge_text(text)
    if not text:
        return None

    match = _CORRECT_FLAG_PATTERN.search(text)
    if not match:
        return None

    return 1.0 if match.group(1).lower() == "true" else 0.0


def _extract_json_object(text: str) -> str:
    text = _clean_judge_text(text)
    if not text:
        return ""

    anchors = []
    for match in _CORRECT_FLAG_PATTERN.finditer(text):
        anchor = text.rfind("{", 0, match.start())
        if anchor != -1:
            anchors.append(anchor)

    if not anchors:
        start = text.find("{")
        if start != -1:
            anchors.append(start)

    for start in anchors:
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        if '"correct"' in candidate.lower():
                            return candidate

    return ""


def llm_judge(question: str, reference: str, candidate: str) -> float:
    prompt = f"""[Question]
{question}

[Reference Final Answer]
{reference}

[Student Final Answer]
{candidate}

Judge: Does the Student Final Answer reach a conclusion equivalent to the Reference Final Answer?
Respond with JSON only: {{"correct": true/false, "reason": "..."}}
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
        "max_tokens": 8192,
    }

    content = ""
    reasoning_content = ""
    try:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        message = result["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        reasoning_content = (message.get("reasoning_content") or "").strip()
        combined = "\n".join([part for part in (content, reasoning_content) if part]).strip()

        # 1. Prefer the simplest possible parse: directly locate `"correct": true/false`.
        for candidate_text in (content, reasoning_content, combined):
            flag = _extract_correct_flag(candidate_text)
            if flag is not None:
                _maybe_log_llm_judge_trace(
                    question=question,
                    reference=reference,
                    candidate=candidate,
                    score=flag,
                    content=content,
                    reasoning_content=reasoning_content,
                    parse_mode="correct_flag",
                    status_code=response.status_code,
                )
                return flag

        # 2. Fall back to extracting a full JSON object if the flag was not found directly.
        json_text = ""
        for candidate_text in (content, reasoning_content, combined):
            json_text = _extract_json_object(candidate_text)
            if json_text:
                break
        if not json_text:
            raise ValueError(
                "No valid JSON object found in judge response. "
                f"content={content[:300]!r} reasoning_content={reasoning_content[:300]!r}"
            )

        data = json.loads(json_text)
        score = 1.0 if data.get("correct", False) else 0.0
        _maybe_log_llm_judge_trace(
            question=question,
            reference=reference,
            candidate=candidate,
            score=score,
            content=content,
            reasoning_content=reasoning_content,
            parse_mode="json_object",
            status_code=response.status_code,
        )
        return score
    except Exception as e:
        status_code = getattr(locals().get("response", None), "status_code", None)
        response_text = getattr(locals().get("response", None), "text", "")
        response_preview = response_text[:1500] if response_text else ""
        _maybe_log_llm_judge_trace(
            question=question,
            reference=reference,
            candidate=candidate,
            score=-1.0,
            content=content,
            reasoning_content=reasoning_content,
            parse_mode="error",
            error=str(e),
            status_code=status_code,
            response_preview=response_preview,
        )
        print(
            "[DeepSeek LLM Judge Warning] "
            f"model={DEEPSEEK_MODEL} status={status_code} error={e} "
            f"response={response_preview}"
        )
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


def _extract_tail_candidate(text: Any, max_chars: int = 500) -> str:
    normalized = _to_text(text).strip()
    if not normalized:
        return ""
    return normalized[-max_chars:]


def _append_debug_log(payload: dict[str, Any]) -> None:
    if not REWARD_DEBUG_LOG or REWARD_DEBUG_LIMIT <= 0:
        return

    log_dir = os.path.dirname(REWARD_DEBUG_LOG)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    lock_path = f"{REWARD_DEBUG_LOG}.lock"
    line = json.dumps(payload, ensure_ascii=False) + "\n"

    try:
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            existing = 0
            if os.path.exists(REWARD_DEBUG_LOG):
                with open(REWARD_DEBUG_LOG, "r", encoding="utf-8") as fh:
                    for existing, _ in enumerate(fh, start=1):
                        pass

            if existing >= REWARD_DEBUG_LIMIT:
                return

            with open(REWARD_DEBUG_LOG, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception as exc:
        print(f"[Reward Debug Warning] Failed to write debug log: {exc}")


def _maybe_log_llm_judge_trace(
    *,
    question: str,
    reference: str,
    candidate: str,
    score: float,
    content: str,
    reasoning_content: str,
    parse_mode: str,
    error: str = "",
    status_code: int | None = None,
    response_preview: str = "",
) -> None:
    _append_debug_log(
        {
            "ts": int(time.time()),
            "log_type": "llm_judge_trace",
            "judge_model": DEEPSEEK_MODEL,
            "status_code": status_code,
            "score": score,
            "parse_mode": parse_mode,
            "error": error,
            "question": question,
            "ground_truth_final": reference,
            "predicted_final": candidate,
            "judge_content": content[:4000],
            "judge_reasoning_content": reasoning_content[:8000],
            "judge_response_preview": response_preview[:4000],
        }
    )


def _maybe_log_reward_case(
    *,
    question: str,
    gt_final: str,
    pred_final: str,
    score: float,
    reason: str,
    used_llm_judge: bool,
    raw_prediction: str,
) -> None:
    _append_debug_log(
        {
            "ts": int(time.time()),
            "score": score,
            "reason": reason,
            "used_llm_judge": used_llm_judge,
            "question": question,
            "ground_truth_final": gt_final,
            "predicted_final": pred_final,
            "raw_prediction_preview": raw_prediction[:4000],
        }
    )


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    pred_text = _to_text(solution_str)
    gt_text = _to_text(ground_truth)
    question = kwargs.get("prompt_str", "Unknown Question")
    if question == "Unknown Question" and isinstance(extra_info, dict):
        question = extra_info.get("prompt", "Unknown Question")

    # Prefer the last non-empty <final> block when present. If the model fails to emit
    # a final block, we fall back to the tail of the raw reasoning text with a capped reward.
    pred_final = _extract_last_nonempty_final(pred_text)
    used_tail_fallback = False
    max_reward_if_correct = 1.0
    pred_candidate = pred_final

    if not pred_candidate:
        pred_candidate = _extract_tail_candidate(pred_text, max_chars=500)
        used_tail_fallback = True
        max_reward_if_correct = 0.8
        if not pred_candidate:
            _maybe_log_reward_case(
                question=question,
                gt_final="",
                pred_final="",
                score=0.0,
                reason="missing_nonempty_final_tag_and_empty_tail_fallback",
                used_llm_judge=False,
                raw_prediction=pred_text,
            )
            return 0.0

    # The GRPO dataset should already store the extracted <final> content as ground_truth.
    # If a tagged string is passed in, still normalize to the final block for safety.
    gt_final = _extract_last_nonempty_final(gt_text) or gt_text.strip()
    if not gt_final:
        _maybe_log_reward_case(
            question=question,
            gt_final="",
            pred_final=pred_candidate,
            score=0.0,
            reason="missing_ground_truth_final",
            used_llm_judge=False,
            raw_prediction=pred_text,
        )
        return 0.0

    # 1. Exact-match fast path on normalized final answers only.
    if _normalize(pred_candidate) == _normalize(gt_final):
        score = max_reward_if_correct
        _maybe_log_reward_case(
            question=question,
            gt_final=gt_final,
            pred_final=pred_candidate,
            score=score,
            reason="normalized_exact_match_tail_fallback" if used_tail_fallback else "normalized_exact_match",
            used_llm_judge=False,
            raw_prediction=pred_text,
        )
        return score

    # 2. LLM-as-judge on final answers only.
    llm_score = llm_judge(question, gt_final, pred_candidate)
    if llm_score != -1.0:
        score = max_reward_if_correct if llm_score > 0.0 else 0.0
        _maybe_log_reward_case(
            question=question,
            gt_final=gt_final,
            pred_final=pred_candidate,
            score=score,
            reason="llm_judge_tail_fallback" if used_tail_fallback else "llm_judge",
            used_llm_judge=True,
            raw_prediction=pred_text,
        )
        return score

    # 3. Conservative fallback: binary exact match only.
    fallback_score = max_reward_if_correct if _normalize(pred_candidate) == _normalize(gt_final) else 0.0
    _maybe_log_reward_case(
        question=question,
        gt_final=gt_final,
        pred_final=pred_candidate,
        score=fallback_score,
        reason="fallback_exact_match_after_llm_failure_tail_fallback"
        if used_tail_fallback
        else "fallback_exact_match_after_llm_failure",
        used_llm_judge=False,
        raw_prediction=pred_text,
    )
    return fallback_score
