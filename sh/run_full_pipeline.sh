#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# End-to-end pipeline:
#   1. preprocess data and inject <think>/<final> instruction
#   2. run 8-GPU vLLM inference with repeated sampling
#   3. run local Qwen3 judge and compute pass@k
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

run_step() {
  local name="$1"
  local script="$2"

  echo ""
  echo "=============================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: ${name}"
  echo "Script: ${script}"
  echo "=============================================================================="

  bash "$script"

  echo "=============================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: ${name}"
  echo "=============================================================================="
}

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

run_step "Step 1 - data preprocessing" "$SCRIPT_DIR/step1_data.sh"
run_step "Step 2 - vLLM generation" "$SCRIPT_DIR/step2.sh"
run_step "Step 3 - local Qwen3 judge" "$SCRIPT_DIR/run_local_judge.sh"

echo ""
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FULL PIPELINE FINISHED"
echo "=============================================================================="
