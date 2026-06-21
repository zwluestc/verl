#!/bin/bash

# 设置 HF 国内镜像源
export HF_ENDPOINT=https://hf-mirror.com

cd /mnt/data/zwl/lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 你的 Qwen3-4B SFT 模型路径
MODEL_PATH="/mnt/data/zwl/models/Qwen3-4B-Base"

echo "=================================================="
echo "使用 vLLM 后端，启动 8 卡并行评测..."
echo "模型: Qwen3-4B-SFT (Edge Only)"
echo "实时日志保存在 eval_vllm_pass1.log"
echo "=================================================="

# 使用 vllm 后端不需要 accelerate launch，直接运行 python 即可
# vLLM 会自动根据 tensor_parallel_size 分配显卡
python -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=8,gpu_memory_utilization=0.8,add_bos_token=True \
    --include_path ./custom_tasks \
    --tasks aime24_sft \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results_aime24_vllm_pass1.json \
    --verbosity INFO 2>&1 | tee eval_vllm_pass1.log