#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" Deepmath_agieval_gaokao_chemistry_mean32 agieval_gaokao_chemistry_mean32
