#!/usr/bin/env bash
set -xeuo pipefail

# === 路径配置 ===
TRAIN_FILES="/mnt/data/zwl/data/medium_edge.jsonl"
MODEL_ID="/mnt/data/zwl/models/Qwen3-0.6B-Base"
CKPT_HOME="/mnt/oss/zwl/checkpoints/qwen3_0.6b_sft"

# === 训练配置 ===
NUM_TRAINERS=1  # 卡数设置为 1
PROJECT_NAME="verl_sft_test"
EXP_NAME="qwen3-0.6b-1gpu"

mkdir -p "${CKPT_HOME}"

# === 启动 torchrun ===
torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=8 \
    data.max_length=2048 \
    data.pad_mode="no_padding" \
    data.truncation="error" \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=65536 \
    model.path="${MODEL_ID}" \
    model.use_remove_padding=True \
    engine="fsdp" \
    engine.strategy="fsdp2" \
    engine.fsdp_size=-1 \
    optim="fsdp" \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style="cosine" \
    trainer.test_freq=-1 \
    trainer.save_freq=500 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode="auto" \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents="[model,optimizer,extra]"