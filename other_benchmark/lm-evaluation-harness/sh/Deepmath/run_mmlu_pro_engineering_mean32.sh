#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" Deepmath_mmlu_pro_engineering_mean32 mmlu_pro_engineering_mean32
