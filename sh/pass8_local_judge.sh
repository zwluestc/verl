#!/bin/bash
# =============================================================================
# 使用本地 Qwen3-8B 模型（8卡并行）判断推理正确性并计算 pass@k
# =============================================================================
# 说明：
#   该脚本调用 judge_by_local_qwen3.py，使用本地部署的 Qwen3-8B 模型
#   直接判断每条问题的 8 次推理是否正确，无需提前提取答案。
#
#   多卡并行机制：
#     - 默认使用 8 张 GPU，每张卡独立加载一份模型
#     - 80 次判断任务（10 题 x 8 次）被均匀分配到 8 张卡上并行执行
#     - 单卡显存需求：bf16 约 16-18GB，若不足会自动 fallback 到 4-bit
#
#   生成长度：
#     - max_new_tokens 默认 12800（给模型充分的思考空间）
#
# 模型路径：/mnt/data/zwl/models/Qwen3-8B
# 输入文件：../output/merged1.jsonl
# 输出文件：../output/qwen3_judge_results.jsonl
# 摘要文件：../output/qwen3_judge_summary.txt
# =============================================================================

set -e

# 切换到项目根目录（假设脚本在 sh/ 目录下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Local Qwen3-8B Judge (8-GPU Parallel)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# 检查模型路径
MODEL_PATH="/mnt/data/zwl/models/Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    echo "Please ensure Qwen3-8B is downloaded to the specified path."
    exit 1
fi

echo "Model path:     $MODEL_PATH"
echo "Input:          output/merged.jsonl"
echo "Output:         output/qwen3_judge_results.jsonl"
echo "Summary:        output/qwen3_judge_summary.txt"
echo "Num questions:  10 (all)"
echo "Num GPUs:       8"
echo "Max tokens:     12800"
echo ""

# -------- 运行 --------
python judge_by_local_qwen3.py \
    --model_path "$MODEL_PATH" \
    --input output/merged.jsonl \
    --output output/qwen3_judge_results.jsonl \
    --summary output/qwen3_judge_summary.txt \
    --num_questions 20000 \
    --num_gpus 8 \
    --max_new_tokens 12800

echo ""
echo "=========================================="
echo "Done! Results saved to output/"
echo "=========================================="
