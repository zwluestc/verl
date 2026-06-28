#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
"${SCRIPT_DIR}/run_mean32_task.sh" WildSci_scibench_mean32 scibench_mean32
