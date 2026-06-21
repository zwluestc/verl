#!/usr/bin/env python3
"""Create data-source-specific GSPO shell scripts from one base script."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SOURCES = ("Metamath", "OpenR1math", "Deepmath", "SciInstruct", "SciRIFF", "WildSci")


def replace_assignment(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*)$", flags=re.MULTILINE)
    replacement = rf"\1${{{name}:-{value}}}"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"Could not find assignment for {name}")
    return new_text


def append_suffix_assignment(text: str, name: str, suffix: str) -> str:
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*)$", flags=re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find assignment for {name}")

    raw_value = match.group(2).strip()
    quote = ""
    value = raw_value
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "'\"":
        quote = raw_value[0]
        value = raw_value[1:-1]

    if not value.endswith(f"_{suffix}"):
        value = f"{value}_{suffix}"

    rendered = f"{match.group(1)}{quote}{value}{quote}"
    return text[: match.start()] + rendered + text[match.end() :]


def build_variant(base_text: str, source: str, data_root: str) -> str:
    source_dir = f"{data_root.rstrip('/')}/{source}"
    text = replace_assignment(base_text, "TRAIN_FILE", f"{source_dir}/train.parquet")
    text = replace_assignment(text, "VAL_FILE", f"{source_dir}/val.parquet")
    text = append_suffix_assignment(text, "OUTPUT_DIR", source)
    text = append_suffix_assignment(text, "TENSORBOARD_DIR", source)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-script",
        type=Path,
        default=Path("sh/instruct-gspo/gspo_4b_mixed_10000_v1.sh"),
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/data/zwl/verl/data/rl/math_10000",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCES),
        choices=SOURCES,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sh/instruct-gspo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_text = args.base_script.read_text(encoding="utf-8")
    stem = args.base_script.stem
    suffix = args.base_script.suffix
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        variant_text = build_variant(base_text, source, args.data_root)
        out_path = args.out_dir / f"{stem}_{source}{suffix}"
        out_path.write_text(variant_text, encoding="utf-8")
        out_path.chmod(args.base_script.stat().st_mode)
        print(out_path)


if __name__ == "__main__":
    main()
