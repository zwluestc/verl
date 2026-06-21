#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from utils import (
    _extract_choice,
    _math_equal,
    _normalize_math_answer,
    doc_to_choice_list,
    doc_to_choice_target,
)


def flatten_once(value):
    if not isinstance(value, list):
        return [value]
    if len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def load_samples(result_dir: Path, task_name: str, sample_file: str | None):
    if sample_file:
        path = Path(sample_file)
    else:
        candidates = sorted(
            result_dir.glob(f"samples_{task_name}_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"未找到 samples 文件: {result_dir}/samples_{task_name}_*.jsonl"
            )
        path = candidates[-1]

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return path, samples


def is_choice_task(task_name: str) -> bool:
    return any(name in task_name for name in ("gpqa", "mmlu_pro", "arc_challenge"))


def target_answer(sample, task_name: str):
    doc = sample.get("doc", {})
    if is_choice_task(task_name):
        try:
            return doc_to_choice_target(doc)
        except Exception:
            target = str(sample.get("target", "")).strip()
            if target.startswith("(") and len(target) >= 2:
                return target[1].upper()
            return target.upper()
    return str(sample.get("target", "")).strip()


def predicted_answers(sample, task_name: str):
    raw_responses = flatten_once(sample.get("resps", []))
    filtered = flatten_once(sample.get("filtered_resps", []))
    responses = raw_responses or filtered

    if is_choice_task(task_name):
        try:
            valid_choices = "ABCDEFGHIJ"[: len(doc_to_choice_list(sample.get("doc", {})))]
        except Exception:
            valid_choices = "ABCDEFGHIJ"
        return [
            {
                "answer": _extract_choice(response, valid_choices),
                "raw": str(response),
            }
            for response in responses
        ]

    return [
        {
            "answer": _normalize_math_answer(response),
            "raw": str(response),
        }
        for response in responses
    ]


def response_correct(prediction: str, target: str, task_name: str) -> bool:
    if is_choice_task(task_name):
        return prediction.upper() == target.upper()
    return _math_equal(prediction, target)


def sample_score(sample):
    value = sample.get("exact_match")
    if value is None and isinstance(sample.get("metrics"), dict):
        value = sample["metrics"].get("exact_match")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0 if bool(value) else 0.0


def write_summary(output_file: Path, task_name: str, sample_path: Path, samples: list[dict]):
    scores = [score for sample in samples if (score := sample_score(sample)) is not None]
    average = sum(scores) / len(scores) if scores else 0.0

    total_correct = 0
    total_predictions = 0
    per_doc = []
    for sample in samples:
        target = target_answer(sample, task_name)
        preds = predicted_answers(sample, task_name)
        rows = []
        for idx, pred in enumerate(preds, 1):
            correct = response_correct(pred["answer"], target, task_name)
            total_correct += int(correct)
            total_predictions += 1
            rows.append((idx, pred["answer"], correct, pred["raw"]))
        per_doc.append((sample, target, rows))

    response_average = (
        total_correct / total_predictions if total_predictions > 0 else average
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Samples file: {sample_path}\n")
        f.write(f"Number of questions: {len(samples)}\n")
        f.write(f"Number of sampled responses: {total_predictions}\n")
        f.write(f"Mean@32 accuracy: {average:.6f}\n")
        f.write(f"Response-level accuracy: {response_average:.6f}\n")
        f.write("\n")

        for sample, target, rows in per_doc:
            doc = sample.get("doc", {})
            question = (
                doc.get("question")
                or doc.get("Problem")
                or doc.get("problem")
                or doc.get("problem_text")
                or doc.get("prompt")
                or ""
            )
            f.write("=" * 100 + "\n")
            f.write(f"doc_id: {sample.get('doc_id')}\n")
            if question:
                f.write(f"question: {question}\n")
            f.write(f"actual_answer: {target}\n")
            f.write(f"doc_mean32_exact_match: {sample_score(sample)}\n")
            f.write("\n")
            for idx, answer, correct, raw in rows:
                f.write(f"[{idx:02d}] inferred_answer: {answer}\n")
                f.write(f"[{idx:02d}] correct: {correct}\n")
                f.write(f"[{idx:02d}] raw_response:\n{raw}\n")
                f.write("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-file", default=None)
    args = parser.parse_args()

    sample_path, samples = load_samples(
        Path(args.result_dir), args.task, args.sample_file
    )
    write_summary(Path(args.output), args.task, sample_path, samples)
    print(f"TXT 汇总已保存至: {args.output}")


if __name__ == "__main__":
    main()
