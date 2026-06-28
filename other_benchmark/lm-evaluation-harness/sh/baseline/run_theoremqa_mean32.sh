#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" baseline_theoremqa_mean32 theoremqa_mean32
