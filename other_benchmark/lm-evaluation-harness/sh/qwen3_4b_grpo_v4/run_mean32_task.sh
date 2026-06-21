#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <task_name> <run_name>"
    exit 1
fi

TASK_NAME="$1"
RUN_NAME="$2"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

WORK_DIR=$(cd "$(dirname "$0")/../.."; pwd)
cd "$WORK_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$WORK_DIR"

MODEL_PATH="${MODEL_PATH:-/mnt/oss/zwl/checkpoints/qwen3_4b_grpo_mixed_2000_v4/global_step_224/huggingface}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-bfloat16}"

RESULT_DIR="${RESULT_DIR:-./results_qwen3_4b_grpo_v4_mean32}"
LOG_DIR="${LOG_DIR:-./logs_qwen3_4b_grpo_v4_mean32}"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

OUTPUT_PATH="${RESULT_DIR}/${RUN_NAME}.json"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
SUMMARY_FILE="${RESULT_DIR}/${RUN_NAME}.txt"

echo "=================================================="
echo "Mean@32 evaluation with vLLM"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "Output: ${OUTPUT_PATH}"
echo "Log: ${LOG_FILE}"
echo "TXT summary: ${SUMMARY_FILE}"
echo "=================================================="

python ./sh/qwen3_4b_grpo_v4/preflight_mean32_dataset.py \
    --include-path ./sh/qwen3_4b_grpo_v4 \
    --task "${TASK_NAME}"

python -m lm_eval --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},add_bos_token=True" \
    --include_path ./sh/qwen3_4b_grpo_v4 \
    --tasks "${TASK_NAME}" \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto \
    --log_samples \
    --output_path "${OUTPUT_PATH}" \
    --verbosity INFO 2>&1 | tee "${LOG_FILE}"

python ./sh/qwen3_4b_grpo_v4/summarize_mean32_results.py \
    --result-dir "${RESULT_DIR}" \
    --task "${TASK_NAME}" \
    --output "${SUMMARY_FILE}"

echo "=================================================="
echo "评测完成。最终 txt 汇总结果在: ${SUMMARY_FILE}"
echo "lm-eval 原始 json/jsonl 结果在: ${RESULT_DIR}"
echo "运行日志在: ${LOG_FILE}"
echo "=================================================="
