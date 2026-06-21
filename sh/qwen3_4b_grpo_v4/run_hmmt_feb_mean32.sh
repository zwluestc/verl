#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" qwen3_4b_grpo_v4_hmmt_feb_mean32 hmmt_feb_mean32
