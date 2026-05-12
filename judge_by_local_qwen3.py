#!/usr/bin/env python3
"""
使用本地部署的 Qwen3-8B 模型（多卡并行）判断推理正确性并计算 pass@k。

核心设计：
1. 从 <final>...</final> 标签中提取最终答案，大幅减少 judge prompt 长度
2. 如果 <final> 缺失，则回退到提取 \boxed{...} 或最后 500 字符
3. 多层 JSON 解析 fallback，防止因模型格式错误导致的误判

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

import multiprocessing as mp

# ========================== 提取逻辑 ==========================


_PLACEHOLDER_ANSWERS = {"", "...", "…", "N/A", "n/a", "None", "none", "null", "NULL"}


def _strip_think(text: str) -> str:
    """去掉模型思考区，保留最终可见回答。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _is_placeholder_answer(text: str) -> bool:
    normalized = text.strip()
    return normalized in _PLACEHOLDER_ANSWERS


def _has_balanced_braces(text: str) -> bool:
    """粗略检查 LaTeX 花括号是否平衡。忽略转义花括号。"""
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _looks_truncated_answer(text: str) -> bool:
    """识别常见的抽取残片，避免把残缺片段送给 judge。"""
    s = text.strip()
    if _is_placeholder_answer(s):
        return True
    if not s:
        return True
    if not _has_balanced_braces(s):
        return True
    suspicious_suffixes = (
        r"\frac",
        r"\frac{",
        r"\sqrt",
        r"\sqrt{",
        r"\begin{aligned",
        r"\begin{cases",
        r"\left",
        r"\boxed{",
    )
    if any(s.endswith(suffix) for suffix in suspicious_suffixes):
        return True
    if r"\begin{aligned" in s and r"\end{aligned" not in s:
        return True
    if r"\begin{cases" in s and r"\end{cases" not in s:
        return True
    return False


def _extract_balanced_command_arg(text: str, command: str) -> str:
    """
    提取最后一个 command{...} 的完整参数，支持嵌套花括号。

    正则 r"\\boxed\\{(.*?)\\}" 会把 \boxed{\frac{a}{b}} 截成
    \frac{a，因此这里改成显式括号配平。
    """
    needle = command + "{"
    start = text.rfind(needle)
    if start == -1:
        return ""

    i = start + len(needle)
    depth = 1
    out = []
    escaped = False
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


def _extract_last_final_tag(text: str) -> str:
    matches = list(re.finditer(r"<final>(.*?)</final>", text, re.DOTALL | re.IGNORECASE))
    for match in reversed(matches):
        candidate = match.group(1).strip()
        if not _is_placeholder_answer(candidate):
            return candidate
    return ""


def _extract_answer_section(text: str) -> str:
    """没有标签/boxed 时，尝试从常见结论标题后截取一小段。"""
    patterns = [
        r"(?:Final Answer|Final answer|Answer|答案|最终答案)\s*[:：]\s*(.+)$",
        r"(?:Therefore|Thus|So|Hence)\s*,?\s*(.+)$",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL))
        for match in reversed(matches):
            candidate = match.group(1).strip()
            if candidate and not _is_placeholder_answer(candidate):
                return candidate[-2000:].strip()
    return ""


def extract_final_answer(text: str) -> str:
    """
    从 <final>...</final> 中提取最终答案。
    如果没有 <final> 标签，则回退到提取最后一个完整的 \boxed{...} 或结论区域。
    """
    if not text:
        return ""

    # 策略 1: 优先提取最后一个非占位 <final>...</final>。
    final_tag = _extract_last_final_tag(text)
    if final_tag:
        return final_tag

    text_no_think = _strip_think(text)

    # 策略 2: 提取最后一个完整 \boxed{...}，支持嵌套 \frac / aligned / cases。
    boxed = _extract_balanced_command_arg(text_no_think, r"\boxed")
    if boxed and not _is_placeholder_answer(boxed):
        return boxed

    # 策略 3: 尝试从 Final Answer / Answer / 因此 等结论区域截取。
    answer_section = _extract_answer_section(text_no_think)
    if answer_section:
        return answer_section

    # 策略 4: 回退到文本最后 2000 字符（通常是结论区域，保留更多上下文）。
    return text_no_think[-2000:].strip()


def choose_candidate_text(record: dict, run_idx: int) -> str:
    """
    选择用于判断的候选文本。

    优先使用 answerX 中已经抽好的最终答案，但如果它为空、占位符或明显残缺，
    就回退到完整 responseX 重新抽取，避免 "..." 或旧抽取残片污染 judge。
    """
    answer = record.get(f"answer{run_idx}", "")
    response = record.get(f"response{run_idx}", "")

    if answer and not _looks_truncated_answer(answer):
        return answer
    return response


def read_jsonl_records(path: Path) -> list:
    """
    读取 JSONL。兼容少量“一个 JSON 对象被意外拆成多行”的情况。
    """
    records = []
    buffer = ""
    start_line = 1
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not buffer:
                start_line = line_no
            buffer += line
            stripped = buffer.strip()
            if not stripped:
                buffer = ""
                continue
            try:
                records.append(json.loads(stripped))
                buffer = ""
            except json.JSONDecodeError:
                # 可能是一个 JSON 对象跨行，继续累积。
                continue

    if buffer.strip():
        raise ValueError(
            f"Failed to parse JSON object starting at line {start_line}: "
            f"{buffer[:200]!r}"
        )

    return records


# ========================== Judge Prompt ==========================

JUDGE_SYSTEM_PROMPT = """You are a strict but fair mathematics and physics professor.
Your task is to judge whether a student's final answer is mathematically equivalent to the reference final answer.

Rules:
1. Compare ONLY the final mathematical conclusions, not the derivation paths.
2. Different notations, equivalent formulas, or different order of terms are acceptable if mathematically equivalent.
3. If the final answer is partially correct but misses critical terms, has wrong signs, or wrong constants, mark as WRONG.
4. If no recognizable final answer is given, mark as WRONG.
5. Output MUST be valid JSON on a single line: {"correct": true/false, "reason": "brief explanation"}
6. Do NOT output thinking tags. Do NOT output markdown code blocks. Output JSON directly.
"""


def build_judge_prompt(question: str, reference: str, candidate: str) -> str:
    """
    构造判断 prompt。使用提取后的最终答案，而非完整原文。
    """
    # 提取 ground truth 的最终答案
    ref_final = extract_final_answer(reference)
    if _looks_truncated_answer(ref_final):
        ref_final = _strip_think(reference)[-2000:].strip()

    # 提取候选答案的最终答案
    cand_final = extract_final_answer(candidate)
    if _looks_truncated_answer(cand_final):
        cand_final = _strip_think(candidate)[-2000:].strip()

    prompt = f"""[Question]
{question}

[Reference Answer (Final)]
{ref_final}

[Student Answer (Final)]
{cand_final}

Judge: Does the Student Answer reach a conclusion equivalent to the Reference Answer?
Think step by step, then respond with JSON only: {{"correct": true/false, "reason": "..."}}
"""
    return prompt


# ========================== 多层 JSON 解析 ==========================


def _parse_judgment(raw_text: str) -> dict:
    """多层 fallback 解析，确保拿到 correct 字段。"""
    cleaned = raw_text.strip()

    # 第 1 层：去掉 <think> 标签及其内容
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # 第 2 层：去掉 markdown code block
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()

    # 第 3 层：尝试完整 JSON 解析
    try:
        # 有时 JSON 前面有额外文本，找第一个 { 和最后一个 }
        if cleaned and cleaned[0] != "{":
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]
        result = json.loads(cleaned)
        if "correct" in result:
            return result
    except Exception:
        pass

    # 第 4 层：正则提取 "correct": true/false（不依赖完整 JSON）
    correct_match = re.search(
        r'"correct"\s*:\s*(true|false)', cleaned, re.IGNORECASE
    )
    if correct_match:
        is_correct = correct_match.group(1).lower() == "true"
        # 尝试提取 reason
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
        reason = reason_match.group(1) if reason_match else "extracted by regex"
        return {"correct": is_correct, "reason": reason}

    # 第 5 层：如果模型输出 "true" 或 "false" 单词
    if re.search(r"\btrue\b", cleaned, re.IGNORECASE):
        return {"correct": True, "reason": "keyword true detected"}
    if re.search(r"\bfalse\b", cleaned, re.IGNORECASE):
        return {"correct": False, "reason": "keyword false detected"}

    # 最终 fallback
    return {
        "correct": False,
        "reason": f"JSON parse failed, raw={raw_text[:200]!r}",
    }


# ========================== 单卡模型加载与推理 ==========================


def load_model_single_gpu(model_path: str, gpu_id: int):
    """在指定 GPU 上加载模型。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

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
    import torch

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
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id
            else tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _parse_judgment(raw_text)


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
    import torch

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
    parser = argparse.ArgumentParser(
        description="Judge reasoning correctness using local Qwen3-8B (multi-GPU)"
    )
    parser.add_argument(
        "--model_path",
        default="/mnt/data/zwl/models/Qwen3-8B",
        help="Path to local Qwen3-8B model",
    )
    parser.add_argument(
        "--input", "-i", default="merged.jsonl", help="Path to merged.jsonl"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="qwen3_judge_results.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary",
        "-s",
        default="qwen3_judge_summary.txt",
        help="Summary text path",
    )
    parser.add_argument(
        "--num_questions",
        "-n",
        type=int,
        default=10,
        help="How many questions to judge (default 10 = all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=12800,
        help="Max tokens for judgment generation (default 12800)",
    )
    parser.add_argument(
        "--num_gpus", "-g", type=int, default=8, help="Number of GPUs to use (default 8)"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # 读取数据。兼容少量 JSON 对象被意外拆成多行的 merged.jsonl。
    records = read_jsonl_records(in_path)

    num_q = min(args.num_questions, len(records))
    total_calls = num_q * 8
    print(f"Loaded {len(records)} questions")
    print(f"Will judge {num_q} questions x 8 runs = {total_calls} calls")
    print(f"Using {args.num_gpus} GPUs, max_new_tokens={args.max_new_tokens}")
    print("MODE: Extract <final> answers to reduce prompt length.")
    print("")

    # 构建所有任务：提取 final 答案后再构造 prompt
    all_tasks = []
    for q_idx in range(num_q):
        rec = records[q_idx]
        question = rec.get("input", "")
        reference = rec.get("output", "")  # 完整参考答案，build_judge_prompt 内部会提取 <final>
        for r_idx in range(1, 9):
            candidate = choose_candidate_text(rec, r_idx)
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
    summary_lines.append("Local Qwen3-8B Multi-GPU Judge Report (with <final> extraction)")
    summary_lines.append("=" * 80)
    summary_lines.append("Mode: Extract <final> answers before judging")

    total_correct = 0
    total_judged = 0

    for q_idx in range(num_q):
        rec = records[q_idx]
        ref_preview = extract_final_answer(rec.get("output", ""))[:200]
        cand_preview = extract_final_answer(choose_candidate_text(rec, 1))[:100]

        q_result = {
            "index": q_idx,
            "question": rec.get("input", "")[:300],
            "judgments": [],
        }
        all_results.append(q_result)

        correct_flags = []

        summary_lines.append(f"\n{'─' * 80}")
        summary_lines.append(f"Question {q_idx}")
        summary_lines.append(f"Reference final: {ref_preview!r}")

        for r_idx in range(1, 9):
            result = results.get((q_idx, r_idx), {"correct": False, "reason": "missing"})
            is_correct = bool(result.get("correct", False))
            reason = result.get("reason", "")

            correct_flags.append(is_correct)
            if is_correct:
                total_correct += 1
            total_judged += 1

            # 候选预览：提取 final 后的前 120 字符
            candidate_full = choose_candidate_text(rec, r_idx)
            cand_snippet = extract_final_answer(candidate_full)[:120].replace("\n", " ")

            q_result["judgments"].append(
                {
                    "run": r_idx,
                    "correct": is_correct,
                    "reason": reason,
                    "candidate_final_preview": cand_snippet,
                }
            )

            marker = "✅" if is_correct else "❌"
            summary_lines.append(f"  {marker} Run{r_idx:02d}: {reason[:120]}")
            summary_lines.append(f"      candidate final: {cand_snippet!r}")

        for k in (1, 2, 4, 8):
            p = compute_pass_at_k(correct_flags, k)
            summary_lines.append(f"      pass@{k} = {p:.4f}")

    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append(
        f"Overall: {total_correct}/{total_judged} correct ({total_correct/max(total_judged,1)*100:.1f}%)"
    )
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
