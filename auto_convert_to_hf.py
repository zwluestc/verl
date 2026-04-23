import torch
import torch.distributed.tensor as dtensor
import os
import glob
import json

# 1. 自动寻找最新的 global_step 文件夹
sft_output_dir = "/mnt/oss/zwl/checkpoints/qwen3_4b_sft"
step_dirs = glob.glob(os.path.join(sft_output_dir, "global_step_*"))

if not step_dirs:
    print("❌ 没有找到任何 global_step_ 文件夹，提取中止。")
    exit(1)

# 按文件夹名称里的数字找出最大/最新的那个
latest_dir = max(step_dirs, key=lambda d: int(d.split('_')[-1]))

# 动态匹配 world_size 的 rank_0 权重文件（解决模型多卡后 model_world_size_4_rank_0.pt 的名称变动）
ckpt_files = glob.glob(os.path.join(latest_dir, "model_world_size_*_rank_0.pt"))
if not ckpt_files:
    print(f"❌ 未在 {latest_dir} 下找到权重点文件 (匹配: model_world_size_*_rank_0.pt)")
    exit(1)

ckpt_path = ckpt_files[0]
out_dir = os.path.join(latest_dir, "huggingface")
out_bin_path = os.path.join(out_dir, "pytorch_model.bin")

print(f"🎯 自动定位到最新 Checkpoint: {ckpt_path}")

# 2. 读取并清洗权重
print(f"📦 正在加载并剥离 FSDP 和 DTensor 外壳...")
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 脱壳处理
if "model" in state_dict:
    state_dict = state_dict["model"]
elif "module" in state_dict:
    state_dict = state_dict["module"]

clean_sd = {}
for k, v in state_dict.items():
    new_k = k.replace("_fsdp_wrapped_module.", "").replace("module.", "")
    # 处理部分可能存在的张量分布结构
    if isinstance(v, dtensor.DTensor):
        v = v.to_local()
    clean_sd[new_k] = v.cpu().clone() if hasattr(v, 'cpu') else v

os.makedirs(out_dir, exist_ok=True)
torch.save(clean_sd, out_bin_path)
print(f"✅ 纯净版标准 HuggingFace 权重已保存至: {out_dir}")

# 3. 自动修改 config.json (解绑 tie_word_embeddings)
config_path = os.path.join(out_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("tie_word_embeddings", True):
        config["tie_word_embeddings"] = False
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("✅ 自动补丁：tie_word_embeddings 已强制设为 false！")
else:
    print(f"⚠️ 提示: {out_dir} 下暂无 config.json。如果你有另外的复制脚本流程，请记得手动设 tie_word_embeddings: false。")
