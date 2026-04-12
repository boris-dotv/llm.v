#!/bin/bash

# SALA hybrid model training pipeline on 8xH100
# Combines 25% sparse attention (InfLLM-V2) + 75% linear attention (SimpleGLA)
# Target: ~2B model with 128K context
# Total time estimate: ~1-2 weeks on 8xH100

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
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

# ============================================================================
# Stage 0: Tokenizer + data download
# ============================================================================
python -m nanochat.dataset -n 16
python -m nanochat.dataset -n 1200 &
python -m scripts.tok_train --max-chars=4000000000 --vocab-size=65536
python -m scripts.tok_eval

# ============================================================================
# Stage 1: Dense pretrain (~31h)
# Standard GPT with d32 (~1.88B params), Chinchilla-optimal 20x ratio
# ============================================================================
echo "=== Stage 1: Dense pretraining ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=32 --target-param-data-ratio=20 --device-batch-size=8 --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval

# ============================================================================
# Stage 2a: HALO conversion (minutes, single-GPU)
# Convert 75% layers to linear attention, keeping first/last as sparse
# ============================================================================
echo "=== Stage 2a: HALO conversion ==="
python -m scripts.halo_convert --model-tag $MODEL_TAG \
    --sparse-layers 0,9,16,17,22,29,30,31 \
    --n-kv-head-sparse 2 --use-output-gate --use-hype

# ============================================================================
# Stage 2b: HALO fine-tune (<1h)
# Train only linear attention layers, 1.3B tokens at 512 context
# ============================================================================
echo "=== Stage 2b: HALO fine-tune ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.halo_train -- \
    --model-tag $MODEL_TAG --max-seq-len 512 --num-tokens 1300000000 --run=$WANDB_RUN

# ============================================================================
# Stage 3: Continual stable-training (~24-48h)
# All params trainable, sparse disabled, 4K context, 30B tokens
# ============================================================================
echo "=== Stage 3: Continual stable-training ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.continual_train -- \
    --model-tag $MODEL_TAG --max-seq-len 4096 --num-tokens 30000000000 \
    --device-batch-size 8 --run=$WANDB_RUN

# ============================================================================
# Stage 4: Long-context adaptation (~40-80h total)
# Progressive context extension with InfLLM-V2 sparse enabled
# ============================================================================

echo "=== Stage 4a: Extend to 32K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source continual --model-tag $MODEL_TAG \
    --phase 32k --max-seq-len 32768 --device-batch-size 2 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0003 --matrix-lr 0.0003 \
    --num-tokens 10000000000 --run=$WANDB_RUN

echo "=== Stage 4b: Extend to 64K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source longctx --model-tag $MODEL_TAG \
    --phase 64k --max-seq-len 65536 --device-batch-size 1 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0002 --matrix-lr 0.0002 \
    --num-tokens 6000000000 --run=$WANDB_RUN

echo "=== Stage 4c: Extend to 128K ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.longctx_train -- \
    --source longctx --model-tag $MODEL_TAG \
    --phase 128k --max-seq-len 131072 --device-batch-size 1 \
    --rope-theta 100000 --gradient-checkpointing \
    --embedding-lr 0.0001 --matrix-lr 0.0001 \
    --final-lr-frac 0.375 \
    --num-tokens 5000000000 --run=$WANDB_RUN

# ============================================================================
# Stage 5: Mid-train + SFT (existing stages)
# ============================================================================
echo "=== Stage 5: Mid-training + SFT ==="
torchrun --standalone --nproc_per_node=$NPROC -m scripts.mid_train -- \
    --model-tag $MODEL_TAG --device-batch-size=8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

echo "=== SALA training complete! ==="
