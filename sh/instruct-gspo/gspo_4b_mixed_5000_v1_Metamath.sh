#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# ==========================================
# 1. 路径配置 (确保指向 HF 格式目录和正确的 RL 数据集)
# ==========================================

#1）输入

BASE_SEARCH_PATH="/mnt/oss/zwl/models/Qwen3-4B-Instruct-2507"
MODEL_PATH=${BASE_SEARCH_PATH}

echo "✅ 已自动定位最新模型路径: ${MODEL_PATH}"

# 默认沿用当前已有 parquet；实际跑 2000 条数据时，请通过 TRAIN_FILE / VAL_FILE
# 显式传入对应的 2000 条数据集路径，避免脚本默认指向一个本地不存在的文件。
TRAIN_FILE=${TRAIN_FILE:-/mnt/data/zwl/verl/data/rl/math_5000/Metamath/train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/data/zwl/verl/data/rl/math_5000/Metamath/val.parquet}

#2）输出

# 保存到对应 checkpoint
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/oss/zwl/checkpoints/qwen3_4b_instruct_2507_gspo_mixed_5000_v1_Metamath}
# 保存对应的 tensorboard
TENSORBOARD_DIR=${TENSORBOARD_DIR:-/mnt/oss/zwl/log/gspo-4B_instruct_2507_mixed_5000_v1_Metamath}

# reward model
REWARD_FN=${REWARD_FN:-${SCRIPT_DIR}/../qwen3_0p6b_deepseek_reward.py}
REWARD_DEBUG_LOG=${REWARD_DEBUG_LOG:-${OUTPUT_DIR}/reward_judge_debug.jsonl}
REWARD_DEBUG_LIMIT=${REWARD_DEBUG_LIMIT:-1000}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
LLM_JUDGE_MIN_INTERVAL=${LLM_JUDGE_MIN_INTERVAL:-0.25}
LLM_JUDGE_MAX_RETRIES=${LLM_JUDGE_MAX_RETRIES:-2}
LLM_JUDGE_MAX_TOKENS=${LLM_JUDGE_MAX_TOKENS:-1024}
LLM_JUDGE_TIMEOUT=${LLM_JUDGE_TIMEOUT:-30}
LLM_JUDGE_ENABLE_THINKING=${LLM_JUDGE_ENABLE_THINKING:-false}

# ==========================================
# 2. 多卡硬件配置
# ==========================================
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# ==========================================
# 3. 内存与 Batch Size 配置
# ==========================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-6}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
ROLLOUT_LOGPROB_MB=${ROLLOUT_LOGPROB_MB:-4}
REF_LOGPROB_MB=${REF_LOGPROB_MB:-4}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-51200}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.40}

LR=${LR:-5e-7}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}

# GSPO 关键配置
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.0003}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.0004}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}

PROJECT_NAME=${PROJECT_NAME:-qwen3_4b_instruct_2507_gspo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_instruct_2507_gspo_v2}

# 每次运行前清空旧输出，避免自动 resume 到历史 checkpoint。
# /mnt/oss is backed by ossfs2; removing the directory itself can fail with
# "Directory not empty" even when no entries are visible. Keep the directory and
# remove only its visible contents.
mkdir -p "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
mkdir -p "${TENSORBOARD_DIR}"
rm -f "${REWARD_DEBUG_LOG}"
cd "${PROJECT_DIR}"

# ==========================================
# 4. 启动 Ray 并执行 GSPO 训练
# ==========================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TENSORBOARD_DIR="${TENSORBOARD_DIR}"
export REWARD_DEBUG_LOG
export REWARD_DEBUG_LIMIT
export LLM_JUDGE_MIN_INTERVAL
export LLM_JUDGE_MAX_RETRIES
export LLM_JUDGE_MAX_TOKENS
export LLM_JUDGE_TIMEOUT
export LLM_JUDGE_ENABLE_THINKING

echo "🚀 Starting fresh GSPO run"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REWARD_DEBUG_LOG=${REWARD_DEBUG_LOG}"
echo "REWARD_DEBUG_LIMIT=${REWARD_DEBUG_LIMIT}"
echo "REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS}"
echo "LLM_JUDGE_MIN_INTERVAL=${LLM_JUDGE_MIN_INTERVAL}"
echo "LLM_JUDGE_MAX_RETRIES=${LLM_JUDGE_MAX_RETRIES}"
echo "LLM_JUDGE_MAX_TOKENS=${LLM_JUDGE_MAX_TOKENS}"
echo "LLM_JUDGE_TIMEOUT=${LLM_JUDGE_TIMEOUT}"
echo "LLM_JUDGE_ENABLE_THINKING=${LLM_JUDGE_ENABLE_THINKING}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "ROLLOUT_N=${ROLLOUT_N}"
echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}"
echo "PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE}"
echo "ROLLOUT_LOGPROB_MB=${ROLLOUT_LOGPROB_MB}"
echo "REF_LOGPROB_MB=${REF_LOGPROB_MB}"
echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "CLIP_RATIO_LOW=${CLIP_RATIO_LOW}"
echo "CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH}"
echo "LOSS_AGG_MODE=${LOSS_AGG_MODE}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MB} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB} \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=${REWARD_NUM_WORKERS} \
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

# ==========================================
# 5. 训练结束后，自动合并 FSDP 权重为 HuggingFace 格式
# ==========================================
echo "🏁 训练结束，开始寻找最新的 Checkpoint 进行合并..."

LATEST_GSPO_STEP_DIR=$(ls -d ${OUTPUT_DIR}/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)

if [ -n "$LATEST_GSPO_STEP_DIR" ]; then
    echo "🔍 找到最新 GSPO Checkpoint: ${LATEST_GSPO_STEP_DIR}"

    ACTOR_DIR="${LATEST_GSPO_STEP_DIR}/actor"
    HF_OUTPUT_DIR="${LATEST_GSPO_STEP_DIR}/huggingface"

    mkdir -p "${HF_OUTPUT_DIR}"

    echo "⚙️ 开始执行 verl.model_merger 将 FSDP 转换为 HF 格式..."
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${ACTOR_DIR}" \
        --target_dir "${HF_OUTPUT_DIR}"

    echo "✅ 转换完成！HuggingFace 格式权重已自动保存在: ${HF_OUTPUT_DIR}"
else
    echo "⚠️ 未找到任何输出的 global_step 文件夹，跳过转换。"
fi
