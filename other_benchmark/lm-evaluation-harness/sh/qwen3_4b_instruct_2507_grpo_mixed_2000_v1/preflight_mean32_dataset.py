#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


TASK_TO_YAML = {
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_aime24_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_aime24_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_aime25_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_aime25_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_amo_bench_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_amo_bench_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_arc_challenge_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_arc_challenge_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_gpqa_diamond_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_gpqa_diamond_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_gpqa_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_gpqa_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_hmmt_feb_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_hmmt_feb_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_mmlu_pro_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_mmlu_pro_mean32.yaml",
    "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_scibench_mean32": "qwen3_4b_instruct_2507_grpo_mixed_2000_v1_scibench_mean32.yaml",
}


class IgnoreFunctionLoader(yaml.SafeLoader):
    pass


def ignore_function(loader, node):
    return loader.construct_scalar(node)


IgnoreFunctionLoader.add_constructor("!function", ignore_function)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--include-path", required=True)
    args = parser.parse_args()

    yaml_name = TASK_TO_YAML.get(args.task)
    if not yaml_name:
        print(f"跳过数据集预检查：未知任务 {args.task}")
        return 0

    yaml_path = Path(args.include_path) / yaml_name
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=IgnoreFunctionLoader)

    dataset_path = cfg.get("dataset_path")
    dataset_name = cfg.get("dataset_name")
    dataset_kwargs = cfg.get("dataset_kwargs") or {}

    if not dataset_path:
        print(f"跳过数据集预检查：{yaml_path} 未配置 dataset_path")
        return 0

    try:
        print(f"数据集预检查: {dataset_path}" + (f" / {dataset_name}" if dataset_name else ""))
        dataset = load_dataset(dataset_path, dataset_name, **dataset_kwargs)
        split = cfg.get("test_split") or cfg.get("validation_split") or cfg.get("training_split")
        if split is not None:
            _ = dataset[split][0]
        print("数据集预检查通过。")
        return 0
    except DatasetNotFoundError as exc:
        message = str(exc)
        print("数据集预检查失败。")
        if "gated dataset" in message or "authenticated" in message:
            print(
                "GPQA 是 Hugging Face gated dataset。请先在 Hugging Face 网页上申请/同意访问，"
                "然后在 DLC 环境中设置 HF_TOKEN 或执行 huggingface-cli login。"
            )
            print("示例：export HF_TOKEN=你的_hf_token")
        print(message)
        return 2
    except Exception as exc:
        print("数据集预检查失败。")
        print(repr(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
