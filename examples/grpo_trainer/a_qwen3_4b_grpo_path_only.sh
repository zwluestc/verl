#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# ==========================================
# 1. 路径配置 (确保指向 HF 格式目录和正确的 RL 数据集)
# ==========================================

#1）输入

BASE_SEARCH_PATH="/mnt/oss/zwl/checkpoints/qwen3_4b_sft_path_only"

# 自动寻找序号最大的 global_step 文件夹
# ls -d 匹配目录, sort -V 按数字版本排序 (100会排在99后面), tail -n 1 取最后一个
LATEST_STEP_DIR=$(ls -d ${BASE_SEARCH_PATH}/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)

if [ -z "$LATEST_STEP_DIR" ]; then
    echo "❌ 错误: 在 ${BASE_SEARCH_PATH} 下没找到任何 global_step_* 文件夹"
    exit 1
fi

# 拼接最终的 HuggingFace 路径
MODEL_PATH="${LATEST_STEP_DIR}/huggingface"

echo "✅ 已自动定位最新模型路径: ${MODEL_PATH}"

#改数据path_only还是edge_only
TRAIN_FILE=${TRAIN_FILE:-/mnt/data/zwl/verl/data/rl/qwen3_4b_grpo_path_only_train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/data/zwl/verl/data/rl/qwen3_4b_grpo_path_only_val.parquet}

#2）输出

#保存到对应cp
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/oss/zwl/checkpoints/qwen3_4b_grpo_path_only}
#保存对应的tensorboard
TENSORBOARD_DIR=${TENSORBOARD_DIR:-/mnt/oss/zwl/log/grpo-4B_path_only}
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

#reward model
REWARD_FN=${REWARD_FN:-${SCRIPT_DIR}/qwen3_0p6b_em_reward.py}

# ==========================================
# 2. 多卡硬件配置
# ==========================================
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8} # 修改为 8 张卡

# ==========================================
# 3. 内存与 Batch Size 配置 (针对 8卡 & 2k数据 调优)
# ==========================================
# 2000条数据，BatchSize为64时，单Epoch约31步
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64} 
# GRPO 的核心组大小，通常为 4~16。提升到 8 可提高强化学习稳定性和探索奖励的精度
ROLLOUT_N=${ROLLOUT_N:-8} 

# 根据 64个Prompt * 8个Rollout = 512 的全局生成样本数，调整以下打分和优化的 Mini/Micro Batch
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}   # 让整体 512 样本切为 4 个 Mini-batch 进行梯度累加更新
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}  # 显存足够可尝试设为 4，OOM则退回 2
ROLLOUT_LOGPROB_MB=${ROLLOUT_LOGPROB_MB:-8}      # 纯前向计算，可设大一点
REF_LOGPROB_MB=${REF_LOGPROB_MB:-8}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.40} # 如果vllm显存吃紧，可微调此项 0.35~0.5 之间

LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}

PROJECT_NAME=${PROJECT_NAME:-qwen3_4b_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_run}

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

# ==========================================
# 4. 启动 Ray 并执行 GRPO 训练
# ==========================================
# 放开所有 8 张卡的可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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