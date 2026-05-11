#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Split JSONL `output` into think/final sections.

The script finds the last occurrence of the Chinese marker ``最终答案`` in the
``output`` field, treats everything before it as the think section, and treats
the text after it as the final answer section.

The rewritten ``output`` field is wrapped as::

    <think>...</think>
    <final>...</final>

If a record does not contain the marker, the whole output is placed inside the
think section and the final section is left empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "最终答案"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"


def split_output(output: str) -> tuple[str, str]:
    marker_index = output.rfind(MARKER)
    if marker_index == -1:
        return output.rstrip(), ""

    think_part = output[:marker_index].rstrip()
    final_part = output[marker_index + len(MARKER) :].lstrip()
    final_part = final_part.lstrip(":： \t\r\n")
    return think_part, final_part


def rewrite_output(output: str) -> str:
    think_part, final_part = split_output(output)
    return f"{THINK_OPEN}{think_part}{THINK_CLOSE}\n{FINAL_OPEN}{final_part}{FINAL_CLOSE}"


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            if "output" not in record or not isinstance(record["output"], str):
                raise ValueError(f"Line {line_number}: missing string output field")

            record["output"] = rewrite_output(record["output"])
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Input JSONL file")
    parser.add_argument("output_path", type=Path, nargs="?", help="Output JSONL file")
    parser.add_argument("--inplace", action="store_true", help="Rewrite the input file in place")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path

    if args.inplace:
        output_path = input_path
    elif args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = input_path.with_name(f"{input_path.stem}_think_final{input_path.suffix}")

    process_file(input_path, output_path)


if __name__ == "__main__":
    main()