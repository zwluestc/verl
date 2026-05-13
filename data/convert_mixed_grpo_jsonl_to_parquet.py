import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = "data/mixed_40.jsonl"
DEFAULT_OUTPUT_DIR = "data/rl"
DEFAULT_OUTPUT_BASENAME = "mixed_40_grpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mixed JSONL into verl GRPO parquet format.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to source JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save parquet files.")
    parser.add_argument(
        "--output-basename",
        default=DEFAULT_OUTPUT_BASENAME,
        help="Base filename used for generated parquet files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Set to 0 to skip train/val split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def extract_last_nonempty_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text or "", flags=re.DOTALL | re.IGNORECASE)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if candidate:
            return candidate
    return ""


def normalize_output_tags(text: str) -> str:
    """
    Keep at most one think block and one final block in a predictable order.
    This is only for metadata/debugging; reward ground_truth still comes from <final>.
    """
    text = text or ""
    think = extract_last_nonempty_tag(text, "think")
    final = extract_last_nonempty_tag(text, "final")

    if final:
        if think:
            return f"<think>\n{think}\n</think>\n<final>\n{final}\n</final>"
        return f"<final>\n{final}\n</final>"

    if think:
        return f"<think>\n{think}\n</think>"

    return text.strip()


def build_prompt(record: dict) -> list[dict]:
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    user_content = instruction + ("\n" + input_text if input_text else "")
    return [{"role": "user", "content": user_content}]


def convert_records(raw_dataset: list[dict], data_source: str) -> list[dict]:
    converted_rows = []
    for idx, item in enumerate(raw_dataset):
        raw_output = str(item.get("output", ""))
        ground_truth = extract_last_nonempty_tag(raw_output, "final")
        if not ground_truth:
            raise ValueError(f"Record {idx} has no non-empty <final>...</final> in output.")
        prompt_messages = build_prompt(item)

        converted_rows.append(
            {
                "uid": idx,
                "data_source": data_source,
                "ability": "reasoning",
                "prompt": prompt_messages,
                "messages": prompt_messages,
                "ground_truth": ground_truth,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index": idx,
                    "difficulty": item.get("difficulty", ""),
                    "view": item.get("view", ""),
                    "prompt": prompt_messages[0]["content"],
                },
                "raw_output": raw_output,
                "normalized_output": normalize_output_tags(raw_output),
            }
        )
    return converted_rows


def save_parquet(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()

    raw_dataset = load_jsonl(args.input_path)
    if not raw_dataset:
        raise ValueError(f"Input dataset is empty: {args.input_path}")

    data_source = Path(args.input_path).stem
    converted_rows = convert_records(raw_dataset, data_source=data_source)

    output_dir = Path(args.output_dir)
    full_path = output_dir / f"{args.output_basename}.parquet"
    save_parquet(converted_rows, full_path)

    print(f"Loaded {len(raw_dataset)} examples from: {args.input_path}")
    print(f"Saved full parquet to: {full_path}")

    if args.val_ratio and args.val_ratio > 0:
        if not 0 < args.val_ratio < 1:
            raise ValueError("--val-ratio must be in (0, 1) when enabled.")
        rng = random.Random(args.seed)
        shuffled = converted_rows[:]
        rng.shuffle(shuffled)

        val_size = max(1, int(round(len(shuffled) * args.val_ratio)))
        val_rows = shuffled[:val_size]
        train_rows = shuffled[val_size:]

        train_path = output_dir / f"{args.output_basename}_train.parquet"
        val_path = output_dir / f"{args.output_basename}_val.parquet"
        save_parquet(train_rows, train_path)
        save_parquet(val_rows, val_path)

        print(f"Saved train parquet ({len(train_rows)} rows) to: {train_path}")
        print(f"Saved val parquet ({len(val_rows)} rows) to: {val_path}")


if __name__ == "__main__":
    main()
