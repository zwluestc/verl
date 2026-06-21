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
    "SciInstruct": {
        "split": "train",
    },
    "SciRIFF": {
        "split": "train",
    },
    "WildSci": {
        "dataset_path": "JustinTX/WildSci",
        "split": "train",
    },
}

SCIENCE_PROMPT_PREFIX = (
    "Please answer the following scientific instruction. Put your final answer "
    "within <final>...</final>."
)
SCIENCE_MCQ_PROMPT_PREFIX = (
    "Please solve the following scientific multiple-choice problem. "
    "Answer with the single correct letter and put it within <final>...</final>."
)


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


def load_local_dataset(input_file: Path) -> Dataset:
    suffix = input_file.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(input_file), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(input_file), split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(input_file), split="train")
    raise ValueError(f"Unsupported input file extension: {input_file}")


def load_source_dataset(
    source: str,
    *,
    input_file: Path | None,
    dataset_path: str | None,
    dataset_name: str | None,
    split: str | None,
) -> Dataset:
    if input_file is not None:
        return load_local_dataset(input_file)

    cfg = SOURCE_CONFIGS[source]
    dataset_path = dataset_path or cfg.get("dataset_path")
    dataset_name = dataset_name or cfg.get("dataset_name")
    split = split or cfg.get("split", "train")
    if not dataset_path:
        raise ValueError(
            f"{source} has no built-in dataset path. Pass --input-file or --dataset-path."
        )
    if dataset_name:
        return load_dataset(dataset_path, dataset_name, split=split)
    return load_dataset(dataset_path, split=split)


def extract_messages_qa(messages: Any) -> tuple[str, str] | None:
    if not isinstance(messages, list):
        return None

    user_parts: list[str] = []
    answer = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = message.get("content") or message.get("value") or ""
        content = normalize_question(content)
        if not content:
            continue
        if role in {"user", "human"}:
            user_parts.append(content)
        elif role in {"assistant", "gpt", "model"}:
            answer = content

    question = "\n\n".join(user_parts).strip()
    if question and answer:
        return question, answer
    return None


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = normalize_question(row.get(key))
        if value:
            return value
    return ""


def format_options(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(
            f"({key}) {value}" for key, value in sorted(options.items())
        )
    if isinstance(options, list):
        lines = []
        for idx, value in enumerate(options):
            label = chr(ord("A") + idx)
            lines.append(f"({label}) {value}")
        return "\n".join(lines)
    return normalize_question(options)


def extract_wildsci_qa(row: dict[str, Any]) -> tuple[str, str] | None:
    question = normalize_question(row.get("question"))
    choices = format_options(row.get("options"))
    answer = clean_answer(row.get("answer"))
    if not question or not choices or not answer:
        return None
    return f"Question: {question}\nChoices:\n{choices}", answer.upper()


def extract_instruction_qa(
    row: dict[str, Any],
    *,
    prompt_column: str | None,
    response_column: str | None,
) -> tuple[str, str] | None:
    if prompt_column or response_column:
        if not prompt_column or not response_column:
            raise ValueError("--prompt-column and --response-column must be used together.")
        question = normalize_question(row.get(prompt_column))
        answer = normalize_question(row.get(response_column))
        return (question, answer) if question and answer else None

    qa = extract_messages_qa(row.get("messages") or row.get("conversations"))
    if qa is not None:
        return qa

    instruction = first_nonempty(
        row,
        ("instruction", "prompt", "question", "query", "content", "input_text"),
    )
    input_text = ""
    if "instruction" in row:
        input_text = first_nonempty(row, ("input", "context"))

    if instruction and input_text:
        question = f"{instruction}\n\n{input_text}"
    else:
        question = instruction or input_text

    answer = first_nonempty(
        row,
        ("summary", "output", "response", "answer", "completion", "target"),
    )
    return (question, answer) if question and answer else None


def extract_qa(
    source: str,
    row: dict[str, Any],
    *,
    prompt_column: str | None,
    response_column: str | None,
) -> tuple[str, str] | None:
    if source == "Metamath":
        question = normalize_question(row.get("query") or row.get("original_question"))
        answer = extract_metamath_answer(row.get("response"))
    elif source == "OpenR1math":
        question = normalize_question(row.get("problem"))
        answer = clean_answer(row.get("answer"))
    elif source == "Deepmath":
        question = normalize_question(row.get("question"))
        answer = clean_answer(row.get("final_answer"))
    elif source in {"SciInstruct", "SciRIFF"}:
        qa = extract_instruction_qa(
            row,
            prompt_column=prompt_column,
            response_column=response_column,
        )
        if qa is None:
            return None
        question, answer = qa
    elif source == "WildSci":
        qa = extract_wildsci_qa(row)
        if qa is None:
            return None
        question, answer = qa
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
    prompt_prefix = (
        SCIENCE_MCQ_PROMPT_PREFIX
        if source == "WildSci"
        else SCIENCE_PROMPT_PREFIX
        if source in {"SciInstruct", "SciRIFF"}
        else PROMPT_PREFIX
    )
    return {
        "data_source": source,
        "prompt": [
            {
                "role": "user",
                "content": f"{prompt_prefix}\n\n{question}",
            }
        ],
        "ability": "science"
        if source in {"SciInstruct", "SciRIFF", "WildSci"}
        else "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "index": index,
            "source_index": source_index,
            "source": source,
            "prompt": question,
        },
    }


def iter_clean_records(
    source: str,
    dataset: Dataset,
    *,
    prompt_column: str | None,
    response_column: str | None,
) -> Iterable[dict[str, Any]]:
    seen_questions: set[str] = set()
    out_index = 0
    for source_index, row in enumerate(dataset):
        qa = extract_qa(
            source,
            row,
            prompt_column=prompt_column,
            response_column=response_column,
        )
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
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Local JSON/JSONL/Parquet/CSV file. Useful for SciInstruct/SciRIFF.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override Hugging Face dataset path, e.g. org/name.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Explicit prompt column for local/HF instruction data.",
    )
    parser.add_argument(
        "--response-column",
        default=None,
        help="Explicit response/reference column for local/HF instruction data.",
    )
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
    dataset = load_source_dataset(
        args.source,
        input_file=args.input_file,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    records = list(
        iter_clean_records(
            args.source,
            dataset,
            prompt_column=args.prompt_column,
            response_column=args.response_column,
        )
    )
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
