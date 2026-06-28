#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" baseline_mmlu_pro_math_mean32 mmlu_pro_math_mean32
