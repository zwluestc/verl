import argparse
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("eval_gspo_10000.py")


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_gspo_10000_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_judge_response_requires_exact_boolean_word():
    module = load_eval_module()

    assert module.parse_judge_response("true") is True
    assert module.parse_judge_response("<think>ignored</think>\nfalse") is False

    for raw in ["not true", "untrue", '{"verdict": "false", "reason": "not true"}']:
        try:
            module.parse_judge_response(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {raw!r}")


def test_unicode_ellipsis_final_is_skipped():
    module = load_eval_module()

    response = "<final>correct</final>\n<final>…</final>"

    assert module.extract_candidate_answer(response) == ("correct", "last_non_empty_final")


def test_exact_match_does_not_call_judge(monkeypatch):
    module = load_eval_module()

    def fail_judge(*args, **kwargs):
        raise AssertionError("judge should not be called for exact matches")

    monkeypatch.setattr(module, "call_deepseek_judge", fail_judge)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    args = argparse.Namespace(
        judge_backend="deepseek-api",
        judge_model_path="unused",
        judge_device_map="unused",
        judge_max_workers=1,
        max_judge_tokens=1,
        log_every=100,
    )
    records = [
        {
            "id": "item-1",
            "input": "question",
            "reference": r"\boxed{answer}",
            "responses": ["  " + r"\boxed{answer}" + "  "],
        }
    ]

    scored = module.score_records(args, records)

    assert scored[0]["scores"][0]["score"] == 1.0
    assert scored[0]["scores"][0]["method"] == "normalized_exact_match"
