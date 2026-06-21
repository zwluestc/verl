#!/bin/bash

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 切换到项目根目录
WORK_DIR=$(cd "$(dirname "$0")/.."; pwd)
cd "$WORK_DIR"
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# 模型路径（请根据实际情况修改）
MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "使用 vLLM 后端，启动 pass@8 评测..."
echo "模型: $MODEL_PATH"
echo "任务: mixed_pass8"
echo "日志保存在 eval_vllm_mixed_pass8.log"
echo "=================================================="

# 运行 lm_eval 评测
# "--include_path ./sh" 指定了包含 mixed_pass8.yaml 的目录
python -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,add_bos_token=True \
    --include_path ./sh \
    --tasks mixed_pass8 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_mixed_pass8.json \
    --verbosity INFO 2>&1 | tee eval_vllm_mixed_pass8.log

echo "=================================================="
echo "生成阶段完成！开始统计 pass@8 指标..."
echo "=================================================="

# 调用 LLM 进行判卷（处理复杂公式和提取推理答案）
python scripts/llm_judge.py

# 调用专门的脚本计算 pass8，根据日志或 samples 文件进行解析统计
python scripts/compute_pass8.py
