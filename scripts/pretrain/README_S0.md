# S0 Dense Pretraining — 608M Dry-Run

Stage S0 = stable-phase dense pretraining. No decay, no mid-training, no
long-context. All 32 layers are `minicpm4` (softmax attention + RoPE).

## Quick Start

### Smoke test (CPU, no GPU, no data download)

```bash
python -m scripts.pretrain.s0_launch --smoke --run dummy
```

### Single GPU dry-run (10B tokens)

```bash
python -m scripts.pretrain.s0_launch \
    --total-tokens 10_000_000_000 \
    --out-dir out/s0_608m_dryrun \
    --run s0-608m-dryrun-v1
```

### Multi-GPU (8x H100), Muon+AdamW (default)

```bash
torchrun --nproc_per_node=8 -m scripts.pretrain.s0_launch \
    --total-tokens 10_000_000_000 \
    --out-dir out/s0_608m_dryrun \
    --run s0-608m-dryrun-v1
```

### Multi-GPU, pure AdamW (FSDP2)

```bash
torchrun --nproc_per_node=8 -m scripts.pretrain.s0_launch \
    --total-tokens 10_000_000_000 \
    --out-dir out/s0_608m_dryrun \
    --run s0-608m-dryrun-v1 \
    --optimizer adamw
```

### With placeholder tokenizer (Track B not yet trained)

```bash
python -m scripts.pretrain.s0_launch \
    --smoke --run dummy \
    --allow-placeholder-tokenizer
```

### Evaluate a checkpoint

```bash
python -m scripts.pretrain.eval_s0 --checkpoint-dir out/s0_608m_dryrun
```

## Expected Performance (8x H100, 10B tokens)

| Metric | Target | Notes |
|---|---|---|
| Wall-clock | ~2-4 hours | Depends on micro-batch size and data streaming speed |
| Final training loss | 2.0 - 2.4 | Cross-entropy on training distribution |
| WikiText-103 PPL | < 20 | Test set perplexity |
| CORE metric | > 0.10 | Aggregate zero-shot accuracy |

### With placeholder Qwen tokenizer (vocab 151K)

When using `--allow-placeholder-tokenizer`, the model has ~785M params (larger
embedding tables) and a mismatched vocab. Expected metrics shift:

| Metric | Target | Notes |
|---|---|---|
| Final training loss | 2.4 - 2.8 | Higher due to vocab entropy mismatch |
| WikiText-103 PPL | < 30 | Degraded — for plumbing validation only |

## PASS Criteria

A dry-run PASSES if ALL of the following hold:

1. Training loss is monotonically decreasing (smoothed, after warmup)
2. No NaN or Inf in loss at any step
3. WikiText-103 test perplexity < 20 (or < 30 with placeholder tokenizer)
4. Checkpoints save and load correctly (verify via `--resume`)

## Model Architecture (608M)

| Field | Value |
|---|---|
| n_embd | 1024 |
| n_layer | 32 |
| n_head | 16 |
| n_kv_head | 1 (GQA 16:1) |
| head_dim | 64 |
| intermediate_size | 4096 (4x hidden) |
| scale_emb | 12.0 (muP) |
| scale_depth | 1.4 |
| dim_model_base | 256 |
| Total params (vocab=65536) | 608M |

## Deviations from SALA Recipe

These are known, intentional deviations from the SALA 9B reference
architecture. Each should be revisited before the 3.5B final run.

### 1. Optimizer: Muon+AdamW (default) vs SALA's pure AdamW

SALA uses standard AdamW with `betas=(0.9, 0.95), weight_decay=0.1,
grad_clip=1.0`. Our default is Muon+AdamW, which replaces AdamW for 2D weight
matrices with an orthogonalized momentum update (Muon), while AdamW handles
embeddings, unembeddings, and 1D parameters.

**Why**: Muon converges faster in the nanochat codebase's existing
infrastructure, reducing dry-run wall-clock time.

**Action for 3.5B**: Benchmark both optimizers on the 608M model. If Muon
provides a clear tokens-to-target-loss advantage, keep it. If not, switch to
pure AdamW for SALA alignment. Use `--optimizer adamw` to test.

### 2. head_dim=64 (SALA uses 128)

At 608M params with depth=32, `head_dim = n_embd / n_head = 1024 / 16 = 64`.
SALA 9B uses `head_dim = 4096 / 32 = 128`.

**Why**: Unavoidable at this model size with depth=32. Reducing depth to
increase head_dim would break SALA's layer-count alignment (needed for HALO
layer-selection in S2).

**Impact**: head_dim is a kernel-level choice, not an architectural one.
FlashAttention handles both 64 and 128. The 3.5B config will naturally have
head_dim=128 (with n_embd=4096, n_head=32).

### 3. Vocab size (may be 151936 with placeholder tokenizer)

The real bilingual tokenizer (Track B) is not yet trained. When using
`--allow-placeholder-tokenizer`, the Qwen2.5-0.5B tokenizer (vocab=151936) is
used as a stand-in. This inflates the model to ~785M params.

**Impact**: Checkpoints produced with the placeholder tokenizer are NOT
transferable. They exercise the training pipeline but produce unusable weights.

### 4. Data mix

The S0 data mix (70% FineWeb-Edu, 15% Stack v2, 10% OpenWebMath, 5% FineWeb-2
Chinese) is a starting point. The SALA paper's exact data mix is not public.
This mix will be refined based on dry-run eval results.
