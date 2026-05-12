#!/usr/bin/env bash
set -euo pipefail

# Parallel vLLM inference launcher for 8 GPUs.
# Adjust PYTHON_BIN if you need to use a virtualenv's python.
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=/mnt/data/zwl/models/Qwen3-8B
INPUT_FILE=/mnt/data/zwl/verl/data/mixed.jsonl
OUTPUT_DIR=/mnt/data/zwl/verl/output
REPEATS=8
WORKERS=8

# Inference parameters
MAX_TOKENS=20480
TEMPERATURE=0.6
TOP_P=0.9

mkdir -p "$OUTPUT_DIR"

PIDS=()
for ((i=0;i<WORKERS;i++)); do
  OUT_PART="$OUTPUT_DIR/part_${i}.jsonl"
  echo "Starting worker $i -> $OUT_PART (GPU $i)"

  # Export CUDA_VISIBLE_DEVICES for this subprocess so vLLM picks the correct GPU.
  (
    export CUDA_VISIBLE_DEVICES=$i
    "$PYTHON_BIN" /mnt/data/zwl/verl/scripts/vllm_infer.py \
      --model "$MODEL_PATH" \
      --input "$INPUT_FILE" \
      --output "$OUT_PART" \
      --gpu-id $i \
      --start-mod $i \
      --mod $WORKERS \
      --repeats $REPEATS \
      --max-tokens $MAX_TOKENS \
      --temperature $TEMPERATURE \
      --top-p $TOP_P
  ) &
  PIDS+=("$!")
done

echo "Launched ${#PIDS[@]} workers; waiting..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "All workers finished. Merging parts into $OUTPUT_DIR/merged.jsonl"
MERGED="$OUTPUT_DIR/merged.jsonl"
rm -f "$MERGED"
for ((i=0;i<WORKERS;i++)); do
  cat "$OUTPUT_DIR/part_${i}.jsonl" >> "$MERGED"
done

echo "Done. Final output: $MERGED"
