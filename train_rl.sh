#!/bin/bash
#SBATCH --job-name=d26_rl
#SBATCH --partition=NV_H100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/public_hw/share/cit_ztyu/cz/nanochat/logs/rl_%j.log
#SBATCH --error=/public_hw/share/cit_ztyu/cz/nanochat/logs/rl_%j.err

cd /public_hw/share/cit_ztyu/cz/nanochat
source .venv/bin/activate

# ===== 关键配置 =====
export NANOCHAT_BASE_DIR=/public_hw/share/cit_ztyu/cz/nanochat
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:/public_hw/share/cit_ztyu/cz/nanochat

# HF 镜像配置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENDPOINT=https://hf-mirror.com

# 数据目录配置
export HF_HOME=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_home
export HF_DATASETS_CACHE=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets
export TRANSFORMERS_CACHE=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_transformers
export HF_HUB_CACHE=/public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_hub

# 确保目录存在
mkdir -p /public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_home
mkdir -p /public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_datasets
mkdir -p /public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_transformers
mkdir -p /public_hw/share/cit_ztyu/cz/nanochat/base_data/hf_hub
# ===================

export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
echo "Using Master Port: $MASTER_PORT"

export WANDB_MODE=offline
export WANDB_PROJECT="nanocontext-rl"
unset TORCH_COMPILE_DISABLE
export TORCHINDUCTOR_MODE=default

MODEL_TAG="d26"
# SFT 训练完成后，模型保存在 chatsft_checkpoints/d26/model_XXXX.pt
# 查看最后的 step 数字，填在这里
MODEL_STEP=700  # 改成实际的 SFT 最终 step

torchrun --nproc_per_node=2 \
    --master_port $MASTER_PORT \
    -- \
    /public_hw/share/cit_ztyu/cz/nanochat/scripts/chat_rl.py \
    --source sft \
    --model-tag $MODEL_TAG \
    --model-step $MODEL_STEP \
    --device-batch-size 8 \
    --examples-per-step 16 \
    --num-samples 16 \
    --num-epochs 1 \
    --eval-every 60 \
    --save-every 60 \
    --run d26_rl_01