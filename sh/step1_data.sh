#!/usr/bin/env bash
set -e

# ==========================================
# Step 1: 数据预处理（添加<think>和<final>标签）
# ==========================================

# 设定项目所在根目录
WORKSPACE_DIR="/mnt/data/zwl/verl"

# 设定需要处理的 JSONL 数据文件路径
# 请根据实际要处理的文件修改此处
INPUT_FILE="${WORKSPACE_DIR}/data/mixed_10.jsonl"

# Python 解释器路径 (根据虚拟环境或全局环境修改)
PYTHON_BIN="python"

echo "==== 开始处理数据 ===="
echo "输入文件: ${INPUT_FILE}"

# 进入项目根目录以确保能正确索引 scripts 文件夹
cd "${WORKSPACE_DIR}"

# 运行 Python 脚本
# --inplace: 直接原地修改文件
# --inject-instruction: 为 instruction 字段注入指定的包含 tag 要求的说明
${PYTHON_BIN} scripts/split_output_to_think_final.py \
    "${INPUT_FILE}" \
    --inplace \
    --inject-instruction

echo "==== 数据预处理完成 ===="
