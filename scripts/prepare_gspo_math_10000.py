#!/usr/bin/env python3
"""Prepare 10k math RL datasets for GSPO training.

The output schema follows the common verl RL parquet format:
data_source, prompt, ability, reward_model, extra_info.
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


PROMPT_PREFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "Metamath": {
        "dataset_path": "meta-math/MetaMathQA",
        "split": "train",
    },
    "OpenR1math": {
        "dataset_path": "open-r1/OpenR1-Math-220k",
        "dataset_name": "default",
        "split": "train",
    },
    "Deepmath": {
        "dataset_path": "zwhe99/DeepMath-103K",
        "split": "train",
    },
}


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    brace_start = text.find("{", idx)
    if brace_start < 0:
        rest = text[idx + len("\\boxed") :].strip()
        return rest.split()[0] if rest else None

    depth = 0
    for pos in range(brace_start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos].strip()
    return None


def clean_answer(answer: Any) -> str:
    text = str(answer).strip()
    if not text:
        return ""

    boxed = last_boxed(text)
    if boxed:
        text = boxed

    text = text.strip().strip("$")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metamath_answer(response: Any) -> str:
    text = str(response or "").strip()
    if not text:
        return ""

    boxed = last_boxed(text)
    if boxed:
        return clean_answer(boxed)

    patterns = [
        r"####\s*(.+?)\s*$",
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)\s*$",
        r"(?:therefore|thus),?\s*(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.splitlines()[-1].strip()
            return clean_answer(candidate)

    return ""


def normalize_question(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def load_source_dataset(source: str) -> Dataset:
    cfg = SOURCE_CONFIGS[source]
    dataset_name = cfg.get("dataset_name")
    if dataset_name:
        return load_dataset(cfg["dataset_path"], dataset_name, split=cfg["split"])
    return load_dataset(cfg["dataset_path"], split=cfg["split"])


def extract_qa(source: str, row: dict[str, Any]) -> tuple[str, str] | None:
    if source == "Metamath":
        question = normalize_question(row.get("query") or row.get("original_question"))
        answer = extract_metamath_answer(row.get("response"))
    elif source == "OpenR1math":
        question = normalize_question(row.get("problem"))
        answer = clean_answer(row.get("answer"))
    elif source == "Deepmath":
        question = normalize_question(row.get("question"))
        answer = clean_answer(row.get("final_answer"))
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not question or not answer:
        return None
    if len(question) < 5 or len(answer) > 512:
        return None
    return question, answer


def build_record(
    *,
    source: str,
    question: str,
    answer: str,
    index: int,
    source_index: int,
) -> dict[str, Any]:
    return {
        "data_source": source,
        "prompt": [
            {
                "role": "user",
                "content": f"{PROMPT_PREFIX}\n\n{question}",
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": index,
            "source_index": source_index,
            "source": source,
        },
    }


def iter_clean_records(source: str, dataset: Dataset) -> Iterable[dict[str, Any]]:
    seen_questions: set[str] = set()
    out_index = 0
    for source_index, row in enumerate(dataset):
        qa = extract_qa(source, row)
        if qa is None:
            continue

        question, answer = qa
        dedup_key = question.lower()
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        yield build_record(
            source=source,
            question=question,
            answer=answer,
            index=out_index,
            source_index=source_index,
        )
        out_index += 1


def sample_records(
    records: list[dict[str, Any]],
    *,
    train_n: int,
    val_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    needed = train_n + val_n
    if len(records) < needed:
        raise ValueError(
            f"Not enough clean records: need {needed}, got {len(records)}."
        )

    rng = random.Random(seed)
    rng.shuffle(records)
    train_records = records[:train_n]
    val_records = records[train_n:needed]
    return train_records, val_records


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCE_CONFIGS), required=True)
    parser.add_argument("--train-n", type=int, default=10_000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory containing train.parquet and val.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_source_dataset(args.source)
    records = list(iter_clean_records(args.source, dataset))
    train_records, val_records = sample_records(
        records,
        train_n=args.train_n,
        val_n=args.val_n,
        seed=args.seed,
    )

    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    write_parquet(train_records, train_path)
    write_parquet(val_records, val_path)

    print(f"source: {args.source}")
    print(f"clean records: {len(records)}")
    print(f"train: {len(train_records)} -> {train_path}")
    print(f"val: {len(val_records)} -> {val_path}")


if __name__ == "__main__":
    main()
