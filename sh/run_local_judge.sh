#!/bin/bash
# =============================================================================
# 使用本地 Qwen3-8B 模型判断推理正确性并计算 pass@k
# =============================================================================
# 说明：
#   该脚本调用 judge_by_local_qwen3.py，使用本地部署的 Qwen3-8B 模型
#   直接判断每条问题的 8 次推理是否正确，无需提前提取答案。
#
# 模型路径：/mnt/data/zwl/models/Qwen3-8B
# 输入文件：../output/merged.jsonl
# 输出文件：../output/qwen3_judge_results.jsonl
# 摘要文件：../output/qwen3_judge_summary.txt
# =============================================================================

set -e

# 切换到项目根目录（假设脚本在 sh/ 目录下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Local Qwen3-8B Judge for pass@k"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# 检查模型路径是否存在
MODEL_PATH="/mnt/data/zwl/models/Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    echo "Please ensure Qwen3-8B is downloaded to the specified path."
    exit 1
fi

echo "Model path: $MODEL_PATH"
echo "Input:      output/merged.jsonl"
echo "Output:     output/qwen3_judge_results.jsonl"
echo "Summary:    output/qwen3_judge_summary.txt"
echo ""

# -------- 配置参数（按需修改） --------
# 要判断的问题数量（默认 2 条用于快速测试，改为 10 可判断全部）
NUM_QUESTIONS=2

# 每次判断生成的最大 token 数
MAX_NEW_TOKENS=256
# --------------------------------------

echo "Judging first $NUM_QUESTIONS questions (8 runs each)..."
echo ""

# 运行判断脚本
python judge_by_local_qwen3.py \
    --model_path "$MODEL_PATH" \
    --input output/merged.jsonl \
    --output output/qwen3_judge_results.jsonl \
    --summary output/qwen3_judge_summary.txt \
    --num_questions "$NUM_QUESTIONS" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo ""
echo "=========================================="
echo "Done! Results saved to output/"
echo "=========================================="
