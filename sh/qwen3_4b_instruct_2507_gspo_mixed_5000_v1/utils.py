import hashlib
import random
import re
from typing import Iterable


CHOICES = "ABCDEFGHIJ"


def _unwrap_repeated_responses(results):
    if len(results) == 1 and isinstance(results[0], list):
        return results[0]
    return results


def _last_boxed(text: str) -> str | None:
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


def _normalize_math_answer(text) -> str:
    text = str(text).strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        text = boxed

    text = text.strip().strip("$")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace(" ", "").replace(",", "")
    return text


def _math_equal(prediction, target) -> bool:
    try:
        from lm_eval.tasks.aime.utils import is_equiv

        return bool(is_equiv(_normalize_math_answer(prediction), str(target)))
    except Exception:
        return _normalize_math_answer(prediction) == _normalize_math_answer(target)


def _extract_choice(text, valid_choices: str) -> str:
    text = str(text).strip()
    boxed = _last_boxed(text)
    if boxed:
        boxed_match = re.search(rf"\b([{valid_choices}])\b", boxed.upper())
        if boxed_match:
            return boxed_match.group(1)

    patterns = [
        rf"(?:answer|答案)\s*(?:is|:|：)?\s*\(?([{valid_choices}])\)?",
        rf"(?:final answer|最终答案)\s*(?:is|:|：)?\s*\(?([{valid_choices}])\)?",
        rf"\(([{valid_choices}])\)",
        rf"\b([{valid_choices}])\b",
    ]
    upper_text = text.upper()
    for pattern in patterns:
        matches = re.findall(pattern, upper_text)
        if matches:
            return matches[-1]
    return ""


def _mean(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _answer_from_doc(doc):
    for key in ("Answer", "answer", "answer_number"):
        if key in doc:
            return doc[key]
    raise KeyError(f"No answer field found in doc keys: {list(doc.keys())}")


def process_aime_mean32(doc, results):
    responses = _unwrap_repeated_responses(results)
    target = _answer_from_doc(doc)
    return {"exact_match": _mean(_math_equal(response, target) for response in responses)}


def process_math_mean32(doc, results):
    responses = _unwrap_repeated_responses(results)
    target = _answer_from_doc(doc)
    return {"exact_match": _mean(_math_equal(response, target) for response in responses)}


def process_choice_mean32(doc, results):
    responses = _unwrap_repeated_responses(results)
    target = str(doc_to_choice_target(doc)).strip().upper()
    valid_choices = CHOICES[: len(doc_to_choice_list(doc))]
    return {
        "exact_match": _mean(
            _extract_choice(response, valid_choices) == target for response in responses
        )
    }


def process_gpqa_docs(dataset):
    def _process_doc(doc):
        choices = [
            _preprocess_gpqa(doc["Incorrect Answer 1"]),
            _preprocess_gpqa(doc["Incorrect Answer 2"]),
            _preprocess_gpqa(doc["Incorrect Answer 3"]),
            _preprocess_gpqa(doc["Correct Answer"]),
        ]
        seed_src = f"{doc['Question']}||{doc['Correct Answer']}"
        seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest(), 16)
        random.Random(seed).shuffle(choices)
        correct = choices.index(_preprocess_gpqa(doc["Correct Answer"]))

        return {
            "question": _preprocess_gpqa(doc["Question"]),
            "choices": choices,
            "answer": CHOICES[correct],
        }

    return dataset.map(_process_doc)


def _preprocess_gpqa(text):
    if text is None:
        return " "
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text)


def doc_to_choice_list(doc):
    if isinstance(doc.get("choices"), dict):
        return list(doc["choices"]["text"])
    if "choices" in doc:
        return list(doc["choices"])
    if "options" in doc:
        return list(doc["options"])
    raise KeyError(f"No choices/options field found in doc keys: {list(doc.keys())}")


def doc_to_choice_target(doc):
    if "answer" in doc:
        answer = doc["answer"]
        if isinstance(answer, int):
            return CHOICES[answer]
        answer = str(answer).strip().upper()
        if answer in CHOICES:
            return answer
        if answer.startswith("(") and len(answer) >= 2 and answer[1] in CHOICES:
            return answer[1]

    if "answerKey" in doc:
        labels = list(doc["choices"]["label"])
        idx = labels.index(doc["answerKey"])
        return CHOICES[idx]

    raise KeyError(f"No supported answer field found in doc keys: {list(doc.keys())}")


def doc_to_mc_text(doc):
    choices = doc_to_choice_list(doc)
    lines = [f"Question: {doc['question']}", "Choices:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({CHOICES[idx]}) {choice}")
    lines.append("Answer with the single correct letter. Put the final letter in \\boxed{}.")
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_arc_text(doc):
    choices = list(doc["choices"]["text"])
    lines = [f"Question: {doc['question']}", "Choices:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({CHOICES[idx]}) {choice}")
    lines.append("Answer with the single correct letter. Put the final letter in \\boxed{}.")
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_mmlu_pro_text(doc):
    lines = [f"Question: {doc['question']}", "Choices:"]
    for idx, option in enumerate(doc["options"]):
        lines.append(f"({CHOICES[idx]}) {option}")
    lines.append("Answer with the single correct letter. Put the final letter in \\boxed{}.")
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_scibench_text(doc):
    unit = str(doc.get("unit") or "").strip()
    lines = [
        f"Question: {doc['problem_text']}",
        "Please solve the scientific problem. Put only the final numerical answer in \\boxed{}.",
    ]
    if unit:
        lines.append(f"The expected unit is: {unit}")
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_amo_bench_text(doc):
    return "\n".join(
        [
            f"Question: {doc['prompt']}",
            "Please solve the problem. Put only the final answer in \\boxed{}.",
            "Answer:",
        ]
    )
