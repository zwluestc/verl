#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

"${SCRIPT_DIR}/run_mmlu_pro_physics_mean32.sh"
"${SCRIPT_DIR}/run_mmlu_pro_chemistry_mean32.sh"
"${SCRIPT_DIR}/run_mmlu_pro_math_mean32.sh"
"${SCRIPT_DIR}/run_mmlu_pro_engineering_mean32.sh"
"${SCRIPT_DIR}/run_sciq_mean32.sh"
"${SCRIPT_DIR}/run_arc_challenge_mean32.sh"
"${SCRIPT_DIR}/run_agieval_gaokao_physics_mean32.sh"
"${SCRIPT_DIR}/run_agieval_gaokao_chemistry_mean32.sh"
