#!/usr/bin/env python3
"""
使用本地部署的 Qwen3-8B 模型判断推理正确性并计算 pass@k。

无需提取答案：直接将【题目 + 参考答案(output) + 候选推理(response)】
送给本地 Qwen3-8B，让模型自己判断候选是否得到了等价正确的结论。

使用方法:
    # 基本用法（判断前 2 条问题的 8 次推理，用于快速测试）
    python judge_by_local_qwen3.py

    # 判断全部 10 条问题（80 次调用）
    python judge_by_local_qwen3.py --num_questions 10

    # 指定其他路径
    python judge_by_local_qwen3.py \
        --model_path /mnt/data/zwl/models/Qwen3-8B \
        --input /mnt/data/zwl/verl/output/merged.jsonl \
        --output /mnt/data/zwl/verl/output/qwen3_judge_results.jsonl

显存要求:
    - bf16: 约 16~18 GB
    - 若显存不足，脚本会自动尝试 4-bit 量化加载 (约 6~8 GB)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ========================== 答案提取工具（复用成熟逻辑） ==========================


def extract_nested_brace(text: str, start_keyword: str = "\\boxed{") -> str:
    """从 text 中提取 start_keyword 开始的嵌套花括号内容。"""
    idx = text.find(start_keyword)
    if idx == -1:
        return ""
    brace_start = idx + len(start_keyword) - 1
    if brace_start >= len(text) or text[brace_start] != "{":
        alt = text.find("{", idx + len(start_keyword) - 1)
        if alt == -1:
            return ""
        brace_start = alt
    depth = 1
    for i in range(brace_start + 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return ""


def extract_final_tag(text: str) -> str:
    if "<final>" not in text:
        return ""
    m = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_after_think(text: str) -> str:
    if "</think>" not in text:
        return ""
    parts = text.split("</think>")
    if len(parts) >= 2:
        after = parts[-1].strip()
        if len(after) > 30:
            return after
    return ""


def extract_answer_from_response(response: str) -> str:
    """按优先级从 response 中提取最终答案（用于展示，不用于判断）。"""
    if not response:
        return ""
    final = extract_final_tag(response)
    if final and len(final) > 5:
        return final
    boxed = extract_nested_brace(response, "\\boxed{")
    if boxed and len(boxed) > 10:
        return boxed
    after_think = extract_after_think(response)
    if after_think:
        boxed2 = extract_nested_brace(after_think, "\\boxed{")
        if boxed2 and len(boxed2) > 10:
            return boxed2
        final2 = extract_final_tag(after_think)
        if final2 and len(final2) > 5:
            return final2
        if len(after_think) > 30:
            return after_think
    tail = response[-800:].strip()
    if len(tail) > 50 and ("$" in tail or "\\" in tail):
        return tail
    return ""


def extract_ground_truth(output: str) -> str:
    """从原始 output 中提取参考答案。"""
    if not output:
        return ""
    final = extract_final_tag(output)
    if final and len(final) > 5:
        return final
    boxed = extract_nested_brace(output, "\\boxed{")
    if boxed and len(boxed) > 10:
        return boxed
    if "<think>" in output:
        parts = output.split("</think>")
        if len(parts) >= 2:
            return parts[-1].strip()
    return output.strip()


# ========================== LLM Judge Prompt ==========================

JUDGE_SYSTEM_PROMPT = """You are a strict but fair mathematics and physics professor.
Your task is to judge whether a student's reasoning and final answer are mathematically equivalent to the reference answer.

Rules:
1. Focus on whether the student's FINAL CONCLUSION is correct and equivalent to the reference.
2. The student may use different notations or derivation paths; these are acceptable if mathematically equivalent.
3. If the student is partially correct but misses critical terms, has wrong signs, or wrong constants, mark as WRONG.
4. If the student gives no recognizable final answer, mark as WRONG.
5. Output MUST be valid JSON: {"correct": true/false, "reason": "brief explanation in English or Chinese"}
"""


def build_judge_prompt(question: str, reference: str, candidate: str) -> str:
    """构造判断 prompt。我们同时提供完整推理给模型，让它自行理解。"""
    # 截断避免过长
    q_trunc = question[:1000]
    ref_trunc = reference[:2000]
    cand_trunc = candidate[:3000]

    prompt = f"""[Question]
{q_trunc}
{"... (truncated)" if len(question) > 1000 else ""}

[Reference Answer]
{ref_trunc}
{"... (truncated)" if len(reference) > 2000 else ""}

[Student Answer]
{cand_trunc}
{"... (truncated)" if len(candidate) > 3000 else ""}

Judge: Does the Student Answer reach a conclusion equivalent to the Reference Answer?
Respond with JSON only: {{"correct": true/false, "reason": "..."}}
"""
    return prompt


# ========================== 本地模型加载与推理 ==========================


def load_model(model_path: str):
    """加载本地 Qwen3-8B，若显存不足自动 fallback 到 4-bit。"""
    print(f"Loading model from: {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 先尝试标准 bf16 加载
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model loaded in bfloat16 mode.")
    except Exception as e:
        print(f"Failed to load in bfloat16: {e}")
        print("Trying 4-bit quantization (requires ~6-8GB VRAM)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("Model loaded in 4-bit mode.")

    model.eval()
    return model, tokenizer


def generate_judgment(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> dict:
    """调用本地模型生成判断结果。"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Qwen3 支持 apply_chat_template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # fallback: 简单拼接
        text = f"System: {JUDGE_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant: "

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 判断任务用 greedy 更稳定
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        )

    # 解码新生成的部分
    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 解析 JSON
    try:
        # 去除可能的 markdown code block
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        result = json.loads(cleaned)
        if "correct" not in result:
            result["correct"] = False
        if "reason" not in result:
            result["reason"] = "missing reason"
        return result
    except Exception as e:
        return {
            "correct": False,
            "reason": f"JSON parse error: {e}, raw={raw_text[:200]!r}",
        }


# ========================== pass@k 计算 ==========================


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
    parser = argparse.ArgumentParser(description="Judge reasoning correctness using local Qwen3-8B")
    parser.add_argument("--model_path", default="/mnt/data/zwl/models/Qwen3-8B", help="Path to local Qwen3-8B model")
    parser.add_argument("--input", "-i", default="merged.jsonl", help="Path to merged.jsonl")
    parser.add_argument("--output", "-o", default="qwen3_judge_results.jsonl", help="Output JSONL path")
    parser.add_argument("--summary", "-s", default="qwen3_judge_summary.txt", help="Summary text path")
    parser.add_argument("--num_questions", "-n", type=int, default=2, help="How many questions to judge (default 2 for test)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for judgment generation")
    parser.add_argument("--no_extract_display", action="store_true", help="Skip displaying extracted answer previews in summary")
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
    print(f"Loaded {len(records)} questions, will judge {num_q} questions x 8 runs = {total_calls} calls")

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    # 开始判断
    all_results = []
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Local Qwen3-8B Judge Report")
    summary_lines.append("=" * 80)

    total_correct = 0
    total_judged = 0

    pbar = tqdm(total=total_calls, desc="Judging")

    for q_idx in range(num_q):
        rec = records[q_idx]
        question = rec.get("input", "")
        reference_full = rec.get("output", "")
        reference_extracted = extract_ground_truth(reference_full)

        q_result = {
            "index": q_idx,
            "question": question[:300],
            "reference_extracted": reference_extracted,
            "judgments": [],
        }
        all_results.append(q_result)

        correct_flags = []

        summary_lines.append(f"\n{'─' * 80}")
        summary_lines.append(f"Question {q_idx}")
        summary_lines.append(f"Reference preview: {reference_extracted[:150]!r}")

        for r_idx in range(1, 9):
            candidate_full = rec.get(f"response{r_idx}", "")
            candidate_extracted = extract_answer_from_response(candidate_full)

            # 构建 prompt：送完整原文给模型判断
            prompt = build_judge_prompt(question, reference_full, candidate_full)
            judgment = generate_judgment(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

            is_correct = bool(judgment.get("correct", False))
            reason = judgment.get("reason", "")

            correct_flags.append(is_correct)
            if is_correct:
                total_correct += 1
            total_judged += 1

            q_result["judgments"].append({
                "run": r_idx,
                "correct": is_correct,
                "reason": reason,
                "candidate_extracted": candidate_extracted if not args.no_extract_display else "",
            })

            marker = "✅" if is_correct else "❌"
            summary_lines.append(
                f"  {marker} Run{r_idx:02d}: {reason[:120]}"
            )
            if not args.no_extract_display:
                summary_lines.append(f"      extracted: {candidate_extracted[:100]!r}")

            pbar.update(1)

        for k in (1, 2, 4, 8):
            p = compute_pass_at_k(correct_flags, k)
            summary_lines.append(f"      pass@{k} = {p:.4f}")

    pbar.close()

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
