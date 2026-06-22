#!/usr/bin/env python3
"""Evaluate GSPO-10000 on our_benchamek/data/benchmark.jsonl.

Pipeline:
1. Generate 32 sampled responses for every benchmark item with vLLM.
2. Score each response by:
   - extracting the last non-empty <final>...</final>, if present;
   - otherwise using the last 1000 characters of the full response;
   - normalized exact match first;
   - falling back to an LLM judge for non-exact matches.
3. Report the mean accuracy for each of the 32 runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import concurrent.futures
import statistics
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path("our_benchamek/data/benchmark.jsonl")
DEFAULT_MODEL = "/mnt/oss/zwl/checkpoints/qwen3_4b_instruct_2507_gspo_mixed_10000_v1"
DEFAULT_OUT = Path("our_benchamek/test/gspo-10000.results.jsonl")
DEFAULT_SUMMARY = Path("our_benchamek/test/gspo-10000.summary.json")
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"

PLACEHOLDERS = {"", "...", "…", "N/A", "n/a", "None", "none", "null", "NULL"}


@dataclass(frozen=True)
class Score:
    value: float
    method: str
    reason: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def is_placeholder(text: str) -> bool:
    return text.strip() in PLACEHOLDERS


def extract_last_final(text: str) -> str:
    matches = list(re.finditer(r"<final>(.*?)</final>", text, re.DOTALL | re.IGNORECASE))
    for match in reversed(matches):
        candidate = match.group(1).strip()
        if not is_placeholder(candidate):
            return candidate
    return ""


def extract_balanced_command_arg(text: str, command: str) -> str:
    """Extract the last balanced LaTeX command argument, e.g. \\boxed{...}."""
    needle = command + "{"
    start = text.rfind(needle)
    if start == -1:
        return ""

    out: list[str] = []
    depth = 1
    escaped = False
    i = start + len(needle)
    while i < len(text):
        ch = text[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return ""


def extract_candidate_answer(response: str) -> tuple[str, str]:
    """Return (candidate, extraction_method) following the requested rules."""
    final = extract_last_final(response)
    if final:
        return final, "last_non_empty_final"
    return response[-1000:].strip(), "last_1000_chars_no_final"


def extract_reference_answer(reference: str) -> str:
    return reference


def normalize_for_exact_match(text: str) -> str:
    return re.sub(r"\s+", " ", strip_think(text)).strip()


def normalized_exact_match(candidate: str, reference: str) -> bool:
    return normalize_for_exact_match(candidate) == normalize_for_exact_match(reference)




JUDGE_SYSTEM_PROMPT = """You are a strict but fair expert judge for symbolic scientific answers.
Decide whether the student's candidate answer is semantically equivalent to the reference answer.

Rules:
1. Judge only the final conclusion, not prose style or derivation path.
2. Equivalent formulas, notation changes, reordered tuple items, and harmless formatting differences are acceptable.
3. Missing required conditions, wrong signs/constants, extra invalid branches, or incomplete multi-part answers are incorrect.
4. The problem statement may contain an answer contract. Enforce it when judging.
5. Reply with a single word: true or false. No other output."""


def build_judge_prompt(question: str, reference: str, candidate: str) -> str:
    return f"""[Problem]
{question}

[Reference answer]
{reference}

[Student candidate answer]
{candidate}

Is the student candidate equivalent to the reference answer? Reply with exactly one word: true or false."""


def parse_judge_response(raw: str) -> bool:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip().lower()
    if cleaned == "true":
        return True
    if cleaned == "false":
        return False
    raise ValueError(f"Judge response must be exactly 'true' or 'false', got: {raw!r}")


def load_vllm_model(model_path: str, tensor_parallel_size: int, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def apply_chat_template(llm: Any, prompt: str) -> str:
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_responses(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    llm = load_vllm_model(args.model_path, args.tensor_parallel_size, args.gpu_memory_utilization)
    prompts = [apply_chat_template(llm, item["input"]) for item in records]
    sampling = SamplingParams(
        n=args.runs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_model_tokens,
    )
    outputs = llm.generate(prompts, sampling)

    by_id: dict[str, list[str]] = {}
    for item, output in zip(records, outputs):
        by_id[item["id"]] = [sample.text for sample in output.outputs]

    generated: list[dict[str, Any]] = []
    for item in records:
        generated.append(
            {
                "id": item["id"],
                "input": item["input"],
                "reference": item["output"],
                "metadata": item.get("metadata", {}),
                "responses": by_id[item["id"]],
            }
        )
    return generated


def call_deepseek_judge(args: argparse.Namespace, prompt: str) -> bool:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=args.deepseek_base_url)
    last_error = None
    for attempt in range(args.judge_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=args.max_judge_tokens,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = response.choices[0].message.content or ""
            return parse_judge_response(raw)
        except Exception as exc:
            last_error = exc
            if attempt >= args.judge_retries:
                break
            time.sleep(args.judge_retry_sleep * (2**attempt))
    raise RuntimeError(f"DeepSeek judge failed after {args.judge_retries + 1} attempts: {last_error}") from last_error


def load_hf_judge(model_path: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def judge_once(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> bool:
    import torch

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {JUDGE_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return parse_judge_response(raw)


def score_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judge_model = judge_tokenizer = None
    if args.judge_backend == "hf":
        judge_model, judge_tokenizer = load_hf_judge(args.judge_model_path, args.judge_device_map)

    if args.judge_backend == "deepseek-api" and not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required when --judge-backend deepseek-api")

    scored: list[dict[str, Any]] = []
    for item_idx, item in enumerate(records):
        reference = extract_reference_answer(item["reference"])
        run_scores: list[dict[str, Any] | None] = [None] * len(item["responses"])
        judge_tasks: list[tuple[int, str, str, str]] = []
        for run_idx, response in enumerate(item["responses"], 1):
            candidate, extraction_method = extract_candidate_answer(response)
            if normalized_exact_match(candidate, reference):
                run_scores[run_idx - 1] = {
                    "run": run_idx,
                    "score": 1.0,
                    "method": "normalized_exact_match",
                    "extraction_method": extraction_method,
                    "candidate_answer": candidate,
                }
                continue
            prompt = build_judge_prompt(item["input"], reference, candidate)
            judge_tasks.append((run_idx, extraction_method, candidate, prompt))

        if judge_tasks and args.judge_backend == "deepseek-api":
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_max_workers) as executor:
                futures = {
                    executor.submit(call_deepseek_judge, args, prompt): (run_idx, extraction_method, candidate)
                    for run_idx, extraction_method, candidate, prompt in judge_tasks
                }
                for future in concurrent.futures.as_completed(futures):
                    run_idx, extraction_method, candidate = futures[future]
                    correct = future.result()
                    run_scores[run_idx - 1] = {
                        "run": run_idx,
                        "score": 1.0 if correct else 0.0,
                        "method": "deepseek_api_judge",
                        "extraction_method": extraction_method,
                        "candidate_answer": candidate,
                    }
        elif judge_tasks:
            for run_idx, extraction_method, candidate, prompt in judge_tasks:
                correct = judge_once(judge_model, judge_tokenizer, prompt, args.max_judge_tokens)
                run_scores[run_idx - 1] = {
                    "run": run_idx,
                    "score": 1.0 if correct else 0.0,
                    "method": "hf_llm_judge",
                    "extraction_method": extraction_method,
                    "candidate_answer": candidate,
                }

        scored.append(
            {
                "id": item["id"],
                "metadata": item.get("metadata", {}),
                "scores": [score for score in run_scores if score is not None],
            }
        )
        if (item_idx + 1) % args.log_every == 0 or item_idx + 1 == len(records):
            print(f"scored {item_idx + 1}/{len(records)}", flush=True)
    return scored


def build_summary(args: argparse.Namespace, scored: list[dict[str, Any]], n_items: int) -> dict[str, Any]:
    by_run: list[list[float]] = [[] for _ in range(args.runs)]
    method_counts: dict[str, int] = {}
    for item in scored:
        for score in item["scores"]:
            by_run[score["run"] - 1].append(float(score["score"]))
            method_counts[score["method"]] = method_counts.get(score["method"], 0) + 1

    run_accuracy = [
        {
            "run": idx + 1,
            "correct": int(sum(values)),
            "total": len(values),
            "accuracy": (sum(values) / len(values)) if values else 0.0,
        }
        for idx, values in enumerate(by_run)
    ]
    all_values = [value for values in by_run for value in values]
    return {
        "name": "gspo-10000",
        "data": str(args.data),
        "num_items": n_items,
        "runs": args.runs,
        "model_path": args.model_path,
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "deepseek_base_url": args.deepseek_base_url if args.judge_backend == "deepseek-api" else None,
        "candidate_rule": "last non-empty <final>...</final>, else last 1000 chars of full response",
        "score_rule": "normalized exact match first, then llm_judge against raw output field; judge errors fail the run",
        "run_accuracy": run_accuracy,
        "mean_accuracy_across_runs": statistics.mean(item["accuracy"] for item in run_accuracy),
        "overall_accuracy": (sum(all_values) / len(all_values)) if all_values else 0.0,
        "method_counts": method_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSPO-10000 on benchmark.jsonl")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--judge-backend", choices=["deepseek-api", "hf"], default="deepseek-api")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--judge-model-path", default=os.environ.get("JUDGE_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--generated", type=Path, default=Path("our_benchamek/test/gspo-10000.generations.jsonl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Use a small prefix for debugging; 0 means all.")
    parser.add_argument("--skip-generate", action="store_true", help="Read --generated instead of running vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-tokens", type=int, default=8192)
    parser.add_argument("--max-judge-tokens", type=int, default=512)
    parser.add_argument("--judge-device-map", default="auto")
    parser.add_argument("--judge-max-workers", type=int, default=16)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_records = read_jsonl(args.data)
    if args.limit:
        source_records = source_records[: args.limit]

    print(f"loaded {len(source_records)} benchmark items from {args.data}")
    print(f"model: {args.model_path}")
    print(f"runs per item: {args.runs}")

    if args.skip_generate:
        generated = read_jsonl(args.generated)
        if args.limit:
            generated = generated[: args.limit]
    else:
        generated = generate_responses(args, source_records)
        write_jsonl(args.generated, generated)
        print(f"wrote generations to {args.generated}")

    for item in generated:
        if len(item.get("responses", [])) != args.runs:
            raise ValueError(f"{item.get('id')} has {len(item.get('responses', []))} responses, expected {args.runs}")

    scored = score_records(args, generated)
    write_jsonl(args.output, scored)
    summary = build_summary(args, scored, len(generated))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote scored results to {args.output}")
    print(f"wrote summary to {args.summary}")
    print(json.dumps(summary["run_accuracy"], ensure_ascii=False, indent=2))
    print(f"mean_accuracy_across_runs={summary['mean_accuracy_across_runs']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
