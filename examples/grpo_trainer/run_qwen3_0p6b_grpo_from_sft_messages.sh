#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# ==========================================
# 1. 路径配置 (确保指向 HF 格式目录和正确的 RL 数据集)
# ==========================================
MODEL_PATH=${MODEL_PATH:-/mnt/oss/zwl/checkpoints/qwen3_0.6b_sft/global_step_78/huggingface}
TRAIN_FILE=${TRAIN_FILE:-/mnt/data/zwl/data/rl/qwen3_0.6b_grpo_train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/data/zwl/data/rl/qwen3_0.6b_grpo_val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/oss/zwl/checkpoints/qwen3_0.6b_grpo}
REWARD_FN=${REWARD_FN:-${SCRIPT_DIR}/qwen3_0p6b_em_reward.py}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-/mnt/data/zwl/verl/log/grpo-0.6B}

# ==========================================
# 2. 多卡硬件配置
# ==========================================
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2} # 明确使用 2 张卡

# ==========================================
# 3. 内存与 Batch Size 配置
# ==========================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
ROLLOUT_LOGPROB_MB=${ROLLOUT_LOGPROB_MB:-4}
REF_LOGPROB_MB=${REF_LOGPROB_MB:-4}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}

LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}

PROJECT_NAME=${PROJECT_NAME:-qwen3_0p6b_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_grpo_from_sft_messages}

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

# ==========================================
# 4. 启动 Ray 并执行 GRPO 训练
# ==========================================
# 锁定前两张卡，防止调度器误抓
export CUDA_VISIBLE_DEVICES=0,1
export TENSORBOARD_DIR="${TENSORBOARD_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.device='cuda' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MB} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB} \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"