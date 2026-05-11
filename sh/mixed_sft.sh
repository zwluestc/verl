#!/usr/bin/env bash
# 开启严格错误处理模式：遇到错误立即停止，未定义变量报错，管道错误报错
set -xeuo pipefail

#模型输出
CKPT_HOME="/mnt/oss/zwl/checkpoints/qwen3_4b_sft_mixed"
TENSORBOARD_DIR=${TENSORBOARD_DIR:-/mnt/oss/zwl/log/sft-4B_mixed}
#模型输入
TRAIN_FILES="/mnt/data/zwl/data/mixed.parquet"
#基座模型
MODEL_ID="/mnt/data/zwl/models/Qwen3-4B-Base"

# 创建输出目录
mkdir -p "${CKPT_HOME}"
mkdir -p "${TENSORBOARD_DIR}"

# 离线运行 wandb
export WANDB_MODE="offline"
export TENSORBOARD_DIR="${TENSORBOARD_DIR}"
# ==========================================
# 1. 清理旧数据 (慎用)
# ==========================================
echo "🧹 正在清理旧的 Checkpoint 和日志..."
# 物理删除旧的 checkpoint 目录和 tensorboard 日志
rm -rf "${CKPT_HOME}"/*
# 重新创建目录确保权限正常
mkdir -p "${CKPT_HOME}"


# ==========================================
# 2. 启动 verl SFT 训练流水线 (4 卡 A100 分布式)
# ==========================================
echo "🚀 开始启动 SFT 分布式训练 (4x GPU)..."

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    -m verl.trainer.sft_trainer \
    data.train_files=$TRAIN_FILES \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=8192 \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=262144 \
    data.ignore_input_ids_mismatch=True \
    model.path=$MODEL_ID \
    model.use_remove_padding=True \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1 \
    optim=fsdp \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.1 \
    "optim.betas=[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    trainer.test_freq=-1 \
    trainer.save_freq=50 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="agenqa_sft" \
    trainer.experiment_name="qwen3-4b-sft-run" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$CKPT_HOME \
    trainer.resume_mode=auto \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents=[model,optimizer,extra]

# ==========================================
# 3. 训练后处理：使用 verl 官方合并工具导出 HF 格式
# ==========================================
echo "======================================"
echo "🎯 训练环节结束，开始自动执行权重格式转换..."

# 1. 自动寻找最新的 global_step 文件夹
LATEST_DIR=$(ls -d ${CKPT_HOME}/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "$LATEST_DIR" ]; then
    echo "❌ 没有找到任何 global_step_ 文件夹，提取中止。"
    exit 1
fi
OUTPUT_HF_DIR="${LATEST_DIR}/huggingface"
mkdir -p "${OUTPUT_HF_DIR}"

# 2. 使用 verl 内置的 model_merger 进行无损拼接
echo "📦 正在加载并合并多卡 FSDP 碎片..."
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LATEST_DIR}" \
    --target_dir "${OUTPUT_HF_DIR}"

# 3. 补充架构文件与纯净的 Tokenizer
echo "📄 正在从基础模型补充架构与 Tokenizer 文件..."
cp -f ${MODEL_ID}/config.json "${OUTPUT_HF_DIR}/" || true
cp -f ${MODEL_ID}/generation_config.json "${OUTPUT_HF_DIR}/" || true
cp -f ${MODEL_ID}/*token*.json "${OUTPUT_HF_DIR}/" || true
cp -f ${MODEL_ID}/*vocab* "${OUTPUT_HF_DIR}/" || true
cp -f ${MODEL_ID}/*merge* "${OUTPUT_HF_DIR}/" || true
cp -f ${MODEL_ID}/*.model "${OUTPUT_HF_DIR}/" || true

# 4. 自动修改 config.json (解绑 tie_word_embeddings)
cat << 'EOF' > patch_config.py
import json
import os
config_path = os.path.join(os.environ["OUTPUT_HF_DIR"], "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("tie_word_embeddings", True):
        config["tie_word_embeddings"] = False
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
EOF
export OUTPUT_HF_DIR
python3 patch_config.py && rm patch_config.py

echo "🎉 SFT 全自动流水线执行完毕！你可以直接将 ${OUTPUT_HF_DIR} 用于 GRPO 或推理了！"
echo "======================================"