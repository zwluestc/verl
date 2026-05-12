#!/usr/bin/env python3
"""
将 qwen3_judge_results.jsonl 中的 pass@8 计算结果合并到 path_only_80.jsonl，
输出为 merged_with_pass.jsonl。
"""
import json
from pathlib import Path


def compute_pass_at_k(correct_flags: list, k: int) -> int:
    """计算 pass@k：k 次推理中做对的次数（整数 0~k）。"""
    return sum(correct_flags[:k])


def main():
    judge_path = Path("output/qwen3_judge_results.jsonl")
    input_path = Path("output/merged.jsonl")
    output_path = Path("output/merged_with_pass.jsonl")

    # 1. 读取 judge 结果，计算 pass@8
    judge_results = {}  # index -> pass@8
    with open(judge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec["index"]
            judgments = rec.get("judgments", [])
            correct_flags = [bool(j.get("correct", False)) for j in judgments]
            pass_at_8 = compute_pass_at_k(correct_flags, k=8)
            judge_results[idx] = pass_at_8

    print(f"Loaded {len(judge_results)} judge results.")

    # 2. 读取输入文件，合并 pass@8
    merged_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # judge 结果中的 index 对应输入文件的行号（0-based）
            if line_idx in judge_results:
                rec["pass@8"] = judge_results[line_idx]
            else:
                rec["pass@8"] = None  # 没有 judge 结果的标记为 null
            merged_records.append(rec)

    # 3. 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in merged_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(merged_records)} records to {output_path}")
    print("Sample pass@8 values:")
    for idx in sorted(judge_results.keys())[:5]:
        print(f"  index {idx}: pass@8 = {judge_results[idx]}")


if __name__ == "__main__":
    main()
