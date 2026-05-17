#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_reward_module():
    module_path = Path(__file__).with_name("qwen3_0p6b_deepseek_reward.py")
    spec = importlib.util.spec_from_file_location("qwen3_0p6b_deepseek_reward", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    reward = _load_reward_module()

    print("Resolved config:")
    print(json.dumps(
        {
            "api_url": reward.DEEPSEEK_API_URL,
            "model": reward.DEEPSEEK_MODEL,
            "api_key_present": bool(reward.DEEPSEEK_API_KEY),
            "api_key_prefix": reward.DEEPSEEK_API_KEY[:8] if reward.DEEPSEEK_API_KEY else "",
        },
        ensure_ascii=False,
        indent=2,
    ))

    question = "Compute 1 + 1."
    reference = "2"
    candidate = "2"

    print("\nRunning llm_judge smoke test...")
    score = reward.llm_judge(question=question, reference=reference, candidate=candidate)
    print(f"llm_judge score: {score}")

    if score == -1.0:
        print("Smoke test failed: API request or response parsing failed.", file=sys.stderr)
        return 1

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
