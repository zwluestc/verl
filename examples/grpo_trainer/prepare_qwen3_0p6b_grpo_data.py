#!/usr/bin/env python3
"""Convert SFT messages parquet into a verl GRPO/RL parquet dataset.

Default input:
  /mnt/data/zwl/data/rl/sft_messages.parquet

Default outputs:
  /mnt/data/zwl/data/rl/qwen3_0.6b_grpo_train.parquet
  /mnt/data/zwl/data/rl/qwen3_0.6b_grpo_val.parquet

Conversion rule:
  - Each row must contain a `messages` column (or fallback `prompt`).
  - The last non-empty assistant message is used as `ground_truth`.
  - All previous messages are used as the RL prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/mnt/data/zwl/data/rl/sft_messages.parquet",
        help="Input SFT parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data/zwl/data/rl",
        help="Directory to save converted GRPO parquet files.",
    )
    parser.add_argument(
        "--prefix",
        default="qwen3_0.6b_grpo",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=64,
        help="Preferred minimum validation samples when dataset is large enough.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    return parser.parse_args()


def to_python_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                return json.loads(text)
            except Exception:
                return value
        return value
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def normalize_role(role: Any) -> str:
    role = str(role).strip().lower()
    mapping = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    return mapping.get(role, role)


def normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise TypeError(f"Unsupported message format: {type(message)}")

    role = message.get("role", message.get("from", message.get("speaker", "user")))
    content = message.get("content", message.get("value", message.get("text", "")))

    if isinstance(content, list):
        content = "\n".join(str(x) for x in content)

    return {
        "role": normalize_role(role),
        "content": "" if content is None else str(content),
    }


def build_rl_record(row: pd.Series, row_idx: int, src_path: str) -> dict[str, Any] | None:
    messages = row.get("messages", row.get("prompt", None))
    messages = to_python_obj(messages)
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    normalized_messages: list[dict[str, str]] = []
    for msg in messages:
        msg = to_python_obj(msg)
        if msg is None:
            continue
        normalized_messages.append(normalize_message(msg))

    if len(normalized_messages) < 2:
        return None

    answer_idx = None
    for idx in range(len(normalized_messages) - 1, -1, -1):
        msg = normalized_messages[idx]
        if msg["role"] == "assistant" and msg["content"].strip():
            answer_idx = idx
            break

    if answer_idx is None or answer_idx == 0:
        return None

    prompt = [m for m in normalized_messages[:answer_idx] if m["content"].strip()]
    ground_truth = normalized_messages[answer_idx]["content"].strip()
    if not prompt or not ground_truth:
        return None

    return {
        "data_source": "custom_sft_exact_match",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "source_file": src_path,
            "row_idx": int(row_idx),
        },
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    train_path = output_dir / f"{args.prefix}_train.parquet"
    val_path = output_dir / f"{args.prefix}_val.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if len(df) == 0:
        raise ValueError(f"Empty parquet file: {input_path}")

    records: list[dict[str, Any]] = []
    skipped = 0
    for row_idx, row in df.iterrows():
        record = build_rl_record(row, row_idx, str(input_path))
        if record is None:
            skipped += 1
            continue
        records.append(record)

    if not records:
        raise ValueError(
            "No usable samples were produced. "
            "Expected each row to contain a `messages` column and end with a non-empty assistant answer."
        )

    out_df = pd.DataFrame(records)

    if len(out_df) == 1:
        train_df = out_df.copy()
        val_df = out_df.copy()
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(out_df))
        suggested_val = max(1, int(len(out_df) * args.val_ratio))
        warm_val = min(args.min_val_samples, max(1, len(out_df) // 10))
        val_size = min(max(suggested_val, warm_val), len(out_df) - 1)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        train_df = out_df.iloc[train_idx].reset_index(drop=True)
        val_df = out_df.iloc[val_idx].reset_index(drop=True)

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"raw rows      : {len(df)}")
    print(f"usable rows   : {len(out_df)}")
    print(f"skipped rows  : {skipped}")
    print(f"train samples : {len(train_df)}")
    print(f"val samples   : {len(val_df)}")
    print(f"train file    : {train_path}")
    print(f"val file      : {val_path}")


if __name__ == "__main__":
    main()