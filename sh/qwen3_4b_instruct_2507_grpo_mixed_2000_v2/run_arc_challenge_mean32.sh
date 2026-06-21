#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" qwen3_4b_instruct_2507_grpo_mixed_2000_v2_arc_challenge_mean32 arc_challenge_mean32
