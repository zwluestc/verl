#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_shared_module():
    shared_path = Path(__file__).resolve().parent.parent / "qwen3_0p6b_deepseek_reward.py"
    spec = importlib.util.spec_from_file_location("shared_qwen3_0p6b_deepseek_reward", shared_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load shared reward module from {shared_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared = _load_shared_module()

for _name in dir(_shared):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_shared, _name)
