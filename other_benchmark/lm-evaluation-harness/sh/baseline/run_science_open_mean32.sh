#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

"${SCRIPT_DIR}/run_qcbench_mean32.sh"
"${SCRIPT_DIR}/run_theoremqa_mean32.sh"
"${SCRIPT_DIR}/run_abench_physics_mean32.sh"
"${SCRIPT_DIR}/run_olympiadbench_physics_mean32.sh"
