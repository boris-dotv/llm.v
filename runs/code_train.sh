#!/bin/bash

# Code-specialized model training pipeline on 8xH100
# 70% code + 30% text pretraining with FIM objective
# Based on Qwen3-Coder-Next recipe adapted for ~2B SALA hybrid model
#
# Total pipeline: data download → tokenizer → pretrain → HALO → continual → longctx → mid-train → SFT → RL

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# Setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

NPROC=8
MODEL_TAG=d32
CODE_DATA_DIR="$NANOCHAT_BASE_DIR/code_data"

# ============================================================================
# Stage 0a: Download pretraining data
# ============================================================================
echo "=== Stage 0a: Downloading code pretraining data ==="
# Download text data (FineWeb-Edu, 30% of training mix)
python -m nanochat.dataset -n 16   # get a few shards to start
python -m nanochat.dataset -n 600 & # download more in background

# Download code data (The Stack v2, 70% of training mix)
python -m scripts.download_code_data \
    --output-dir $CODE_DATA_DIR \
    --shard-size 100000

# ============================================================================
# Stage 0b: Download SFT/RL datasets
# ============================================================================
echo "=== Stage 0b: Downloading SFT/RL datasets ==="
python -m scripts.download_post_data

# ============================================================================
# Stage 0c: Train tokenizer on code+text mix
# ============================================================================
echo "=== Stage 0c: Training tokenizer ==="
python -m scripts.tok_train \
    --max-chars=6000000000 \
    --vocab-size=65536 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7

python -m scripts.tok_eval

# ============================================================================
# Stage 1: Dense pretrain
# Code-heavy: 70% code with FIM + 30% FineWeb-Edu text
# ============================================================================
echo "=== Stage 1: Dense pretraining (code-heavy) ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=32 --target-param-data-ratio=20 \
    --device-batch-size=8 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7 \
    --fim-rate=0.5 \
    --spm-rate=0.5 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval

# ============================================================================
# Stage 2a: HALO conversion (dense → hybrid)
# ============================================================================
echo "=== Stage 2a: HALO conversion ==="
python -m scripts.halo_convert --model-tag $MODEL_TAG \
    --sparse-layers 0,9,16,17,22,29,30,31 \
    --n-kv-head-sparse 2 --use-output-gate --use-hype

# ============================================================================
# Stage 2b: HALO fine-tune (<1h)
# ============================================================================
echo "=== Stage 2b: HALO fine-tune ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.halo_train -- \
    --model-tag $MODEL_TAG --max-seq-len 512 --num-tokens 1300000000 --run=$WANDB_RUN

# ============================================================================
# Stage 3: Continual stable-training with code data
# ============================================================================
echo "=== Stage 3: Continual stable-training (code-heavy) ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.continual_train -- \
    --model-tag $MODEL_TAG --max-seq-len 4096 --num-tokens 30000000000 \
    --device-batch-size 8 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7 --fim-rate=0.5 \
    --run=$WANDB_RUN

# ============================================================================
# Stage 4: Long-context adaptation (32K → 64K → 128K) with code data
# ============================================================================
echo "=== Stage 4a: Extend to 32K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source continual --model-tag $MODEL_TAG \
    --phase 32k --max-seq-len 32768 --device-batch-size 2 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0003 --matrix-lr 0.0003 \
    --num-tokens 10000000000 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7 --fim-rate=0.5 \
    --run=$WANDB_RUN

echo "=== Stage 4b: Extend to 64K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source longctx --model-tag $MODEL_TAG \
    --phase 64k --max-seq-len 65536 --device-batch-size 1 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0002 --matrix-lr 0.0002 \
    --num-tokens 6000000000 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7 --fim-rate=0.5 \
    --run=$WANDB_RUN

echo "=== Stage 4c: Extend to 128K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source longctx --model-tag $MODEL_TAG \
    --phase 128k --max-seq-len 131072 --device-batch-size 1 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0001 --matrix-lr 0.0001 \
    --final-lr-frac 0.375 \
    --num-tokens 5000000000 \
    --code-data-dir=$CODE_DATA_DIR \
    --code-weight=0.7 --fim-rate=0.5 \
    --run=$WANDB_RUN

# ============================================================================
# Stage 5: Code mid-training
# ============================================================================
echo "=== Stage 5: Code mid-training ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.mid_train -- \
    --model-tag $MODEL_TAG --device-batch-size=8 --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.code_eval -- -i mid

# ============================================================================
# Stage 6: Code SFT
# ============================================================================
echo "=== Stage 6: Code SFT ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.code_eval -- -i sft

# ============================================================================
# Stage 7: Code RL (execution-verified)
# ============================================================================
echo "=== Stage 7: Code RL ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.code_rl -- \
    --max-new-tokens=1024 --num-samples=8 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.code_eval -- -i rl

echo "=== Code model training complete! ==="
