#!/usr/bin/env python3
"""
使用本地部署的 Qwen3-8B 模型（多卡并行）判断推理正确性并计算 pass@k。

核心设计：不做任何答案提取，直接将【题目 + 完整参考答案(output) + 完整候选推理(response)】
送给本地 Qwen3-8B，让模型自己判断候选的推理过程和最终结论是否与参考等价。

多卡并行说明：
    脚本会将 80 次判断任务（10 题 x 8 次）均匀分配到 N 张 GPU 上，
    每张卡独立加载一份模型，并行处理不同的请求（数据并行）。

显存要求（每卡）：
    - bf16 模式: ~16-18 GB
    - 4-bit 量化: ~6-8 GB

使用方法:
    # 默认：8卡并行，判断全部 10 条问题（80 次调用）
    python judge_by_local_qwen3.py

    # 指定其他路径
    python judge_by_local_qwen3.py \
        --model_path /mnt/data/zwl/models/Qwen3-8B \
        --input /mnt/data/zwl/verl/output/merged.jsonl \
        --output /mnt/data/zwl/verl/output/qwen3_judge_results.jsonl

    # 只用 4 张卡
    python judge_by_local_qwen3.py --num_gpus 4
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

# ========================== Judge Prompt ==========================

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


def build_judge_prompt(question: str, reference: str, candidate: str) -> str:
    """
    构造判断 prompt。不做任何提取、不做任何截断，直接使用完整原文。
    """
    prompt = f"""[Question]
{question}

[Reference Answer]
{reference}

[Student Answer]
{candidate}

Judge: Does the Student Answer reach a conclusion equivalent to the Reference Answer?
Think step by step, then respond with JSON only: {{"correct": true/false, "reason": "..."}}
"""
    return prompt


# ========================== 单卡模型加载与推理 ==========================


def load_model_single_gpu(model_path: str, gpu_id: int):
    """在指定 GPU 上加载模型。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.cuda.set_device(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": gpu_id},
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] bfloat16 failed: {e}, trying 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": gpu_id},
        )

    model.eval()
    return model, tokenizer


def generate_judgment(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = f"System: {JUDGE_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant: "

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    try:
        cleaned = raw_text.strip()

        # 1. 去掉 <think>...</think> 及其内容
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # 2. 去掉 markdown code block
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # 3. 有时候 JSON 前面有额外文本，尝试找第一个 { 和最后一个 }
        if cleaned and cleaned[0] != "{":
            start = cleaned.find("{")
            if start != -1:
                end = cleaned.rfind("}")
                if end != -1:
                    cleaned = cleaned[start:end+1]

        result = json.loads(cleaned)
        if "correct" not in result:
            result["correct"] = False
        if "reason" not in result:
            result["reason"] = "missing reason"
        return result
    except Exception as e:
        return {
            "correct": False,
            "reason": f"JSON parse error: {e}, raw={raw_text[:300]!r}",
        }


# ========================== 多卡 Worker ==========================


def worker_fn(gpu_id, task_list, model_path, max_new_tokens, result_queue):
    """
    每个 worker 独占一张 GPU，处理分配到的任务列表。
    task_list: [(q_idx, r_idx, prompt), ...]
    """
    print(f"[Worker GPU {gpu_id}] Loading model...")
    model, tokenizer = load_model_single_gpu(model_path, gpu_id)
    print(f"[Worker GPU {gpu_id}] Ready, processing {len(task_list)} tasks.")

    for q_idx, r_idx, prompt in task_list:
        result = generate_judgment(model, tokenizer, prompt, max_new_tokens)
        result_queue.put((q_idx, r_idx, result))

    del model
    torch.cuda.empty_cache()
    print(f"[Worker GPU {gpu_id}] Done.")


# ========================== pass@k ==========================


def compute_pass_at_k(correct_flags: list, k: int) -> float:
    n = len(correct_flags)
    c = sum(correct_flags)
    if k > n:
        return 0.0
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


# ========================== 主函数 ==========================


def main():
    parser = argparse.ArgumentParser(description="Judge reasoning correctness using local Qwen3-8B (multi-GPU)")
    parser.add_argument("--model_path", default="/mnt/data/zwl/models/Qwen3-8B", help="Path to local Qwen3-8B model")
    parser.add_argument("--input", "-i", default="merged.jsonl", help="Path to merged.jsonl")
    parser.add_argument("--output", "-o", default="qwen3_judge_results.jsonl", help="Output JSONL path")
    parser.add_argument("--summary", "-s", default="qwen3_judge_summary.txt", help="Summary text path")
    parser.add_argument("--num_questions", "-n", type=int, default=10, help="How many questions to judge (default 10 = all)")
    parser.add_argument("--max_new_tokens", type=int, default=12800, help="Max tokens for judgment generation (default 12800)")
    parser.add_argument("--num_gpus", "-g", type=int, default=8, help="Number of GPUs to use (default 8)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # 读取数据
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    num_q = min(args.num_questions, len(records))
    total_calls = num_q * 8
    print(f"Loaded {len(records)} questions")
    print(f"Will judge {num_q} questions x 8 runs = {total_calls} calls")
    print(f"Using {args.num_gpus} GPUs, max_new_tokens={args.max_new_tokens}")
    print("NOTE: No answer extraction -- sending FULL response text to judge model.")
    print("")

    # 构建所有任务：直接使用完整 output 和 response，不做任何提取
    all_tasks = []
    for q_idx in range(num_q):
        rec = records[q_idx]
        question = rec.get("input", "")
        reference = rec.get("output", "")        # 完整参考答案
        for r_idx in range(1, 9):
            candidate = rec.get(f"response{r_idx}", "")  # 完整候选推理
            prompt = build_judge_prompt(question, reference, candidate)
            all_tasks.append((q_idx, r_idx, prompt))

    # 按 GPU 数量切分任务
    num_gpus = args.num_gpus
    task_chunks = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        task_chunks[i % num_gpus].append(task)

    # 启动多进程
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, task_chunks[gpu_id], args.model_path, args.max_new_tokens, result_queue),
        )
        p.start()
        processes.append(p)

    # 收集结果
    results = {}
    expected = len(all_tasks)
    print(f"Collecting results from {num_gpus} workers...")
    for i in range(expected):
        q_idx, r_idx, result = result_queue.get()
        results[(q_idx, r_idx)] = result
        if (i + 1) % 10 == 0 or (i + 1) == expected:
            print(f"  Progress: {i + 1}/{expected}")

    for p in processes:
        p.join()

    print(f"\nAll {expected} judgments collected.")

    # 汇总
    all_results = []
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Local Qwen3-8B Multi-GPU Judge Report")
    summary_lines.append("=" * 80)
    summary_lines.append("Mode: FULL response comparison (no answer extraction)")

    total_correct = 0
    total_judged = 0

    for q_idx in range(num_q):
        rec = records[q_idx]
        ref_preview = rec.get("output", "")[:200]   # 仅用于展示
        cand_preview = rec.get("response1", "")[:100]  # 仅用于展示

        q_result = {
            "index": q_idx,
            "question": rec.get("input", "")[:300],
            "judgments": [],
        }
        all_results.append(q_result)

        correct_flags = []

        summary_lines.append(f"\n{'─' * 80}")
        summary_lines.append(f"Question {q_idx}")
        summary_lines.append(f"Reference preview: {ref_preview!r}")

        for r_idx in range(1, 9):
            result = results.get((q_idx, r_idx), {"correct": False, "reason": "missing"})
            is_correct = bool(result.get("correct", False))
            reason = result.get("reason", "")

            correct_flags.append(is_correct)
            if is_correct:
                total_correct += 1
            total_judged += 1

            # 候选预览：直接截断前 120 字符，不提取
            candidate_full = rec.get(f"response{r_idx}", "")
            cand_snippet = candidate_full[:120].replace("\n", " ")

            q_result["judgments"].append({
                "run": r_idx,
                "correct": is_correct,
                "reason": reason,
                "candidate_preview": cand_snippet,
            })

            marker = "✅" if is_correct else "❌"
            summary_lines.append(f"  {marker} Run{r_idx:02d}: {reason[:120]}")
            summary_lines.append(f"      candidate preview: {cand_snippet!r}")

        for k in (1, 2, 4, 8):
            p = compute_pass_at_k(correct_flags, k)
            summary_lines.append(f"      pass@{k} = {p:.4f}")

    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append(f"Overall: {total_correct}/{total_judged} correct ({total_correct/max(total_judged,1)*100:.1f}%)")
    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # 写入结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nWrote results to: {out_path}")

    sum_path = Path(args.summary)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Wrote summary to: {sum_path}")


if __name__ == "__main__":
    main()
