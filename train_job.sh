#!/bin/bash
#SBATCH --job-name=d26_2
#SBATCH --partition=NV_H100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/public_hw/share/cit_ztyu/cz/nanochat/logs/train_%j.log
#SBATCH --error=/public_hw/share/cit_ztyu/cz/nanochat/logs/train_%j.err

cd /public_hw/share/cit_ztyu/cz/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR=/public_hw/share/cit_ztyu/cz/nanochat
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:/public_hw/share/cit_ztyu/cz/nanochat

# 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
# wandb login
# wandb sync wandb/offline-run-20260125_191256-xsauou2o

export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
echo "Using Master Port: $MASTER_PORT"

export WANDB_MODE=offline 
export WANDB_PROJECT="nanocontext"
# 关键：禁用 torch.compile 避免编译问题
# export TORCH_COMPILE_DISABLE=1
unset TORCH_COMPILE_DISABLE
export TORCHINDUCTOR_MODE=default

RESUME_ARG="--resume-from-step 76500"

torchrun --nproc_per_node=2 \
    --master_port $MASTER_PORT \
    -- \
    /public_hw/share/cit_ztyu/cz/nanochat/scripts/base_train.py \
    --depth 26 \
    --max-seq-len 2048 \
    --device-batch-size 8 \
    --num-iterations 153600 \
    --eval-every 200 \
    --save-every 500 \
    --run d26_run_01 \
    $RESUME_ARG