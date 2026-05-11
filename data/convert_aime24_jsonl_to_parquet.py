import argparse
import json
import os

import pandas as pd


DEFAULT_INPUT_PATH = "/mnt/data/zwl/verl/data/test/aime_2024.jsonl"
DEFAULT_OUTPUT_PATH = "/mnt/data/zwl/verl/data/output/aime24_input.parquet"
DEFAULT_USER_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AIME 2024 jsonl into verl generation input parquet.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to the source aime_2024.jsonl file.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save the converted parquet file.")
    parser.add_argument(
        "--user-suffix",
        default=DEFAULT_USER_SUFFIX,
        help="Optional suffix appended to the user problem prompt.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def main() -> None:
    args = parse_args()

    raw_dataset = load_jsonl(args.input_path)
    if not raw_dataset:
        raise ValueError(f"Input dataset is empty: {args.input_path}")

    converted_rows = []
    for idx, item in enumerate(raw_dataset):
        problem = str(item["problem"]).strip()
        ground_truth = str(item["ground_truth"]).strip()
        source = item.get("source", "unknown")

        user_content = problem
        if args.user_suffix and args.user_suffix.strip() not in user_content:
            user_content += args.user_suffix

        messages = [{"role": "user", "content": user_content}]

        converted_rows.append(
            {
                "uid": idx,
                "data_source": "aime24",
                "prompt": messages,
                "messages": messages,
                "problem": problem,
                "ground_truth": ground_truth,
                "reward_model": {"ground_truth": ground_truth},
                "source": source,
            }
        )

    df = pd.DataFrame(converted_rows)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(args.output_path, index=False)

    print(f"Loaded {len(raw_dataset)} AIME 2024 examples from: {args.input_path}")
    print(f"Saved converted parquet to: {args.output_path}")


if __name__ == "__main__":
    main()