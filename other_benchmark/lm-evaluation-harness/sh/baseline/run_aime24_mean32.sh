#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" baseline_aime24_mean32 aime24_mean32
