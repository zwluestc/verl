#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Please set DEEPSEEK_API_KEY before running this script." >&2
  exit 1
fi

cd "$(dirname "$0")/../.."

python benchamek/test/eval_gspo_10000.py \
  --data benchamek/data/benchmark.jsonl \
  --model-path /mnt/oss/zwl/models/Qwen3-4B-Instruct-2507 \
  --runs 32 \
  --tensor-parallel-size 4 \
  --judge-backend deepseek-api \
  --judge-model deepseek-v4-flash \
  --deepseek-base-url https://www.dmxapi.cn/v1/chat/completions \
  --judge-max-workers 16 \
  --output benchamek/test/instruct-baseline.results.jsonl \
  --summary benchamek/test/instruct-baseline.summary.json
