# Manual SALA Implementation Guide

> **Purpose**: Implement the SALA hybrid attention model (sparse + linear) by hand, step by step, to deeply understand every piece of the training pipeline.
>
> **Rule**: All code in this directory is written by YOU. AI only modifies this `instruction.md` file.

---

## Overview

You will build a SALA hybrid model on top of the existing nanochat codebase. The work is split into 7 milestones. Each milestone is self-contained — you can verify it works before moving on.

**Final goal**: A ~2B model with 8 sparse + 24 linear attention layers, trained from dense to hybrid, with 128K context support.

---

## Directory Structure (you will create these)

```
manual_impl/
├── instruction.md          ← this file (AI-editable only)
├── 01_linear_attention.py  ← Milestone 1: standalone lightning attention module
├── 02_sparse_attention.py  ← Milestone 2: InfLLM-V2 wrapper
├── 03_hybrid_gpt.py        ← Milestone 3: modified GPT with hybrid layers
├── 04_halo_convert.py      ← Milestone 4: dense→hybrid weight conversion
├── 05_halo_train.py        ← Milestone 5: freeze-all-but-linear training
├── 06_continual_train.py   ← Milestone 6: full-param continual training
├── 07_longctx_train.py     ← Milestone 7: long-context with sparse enabled
└── tests/
    └── test_hybrid.py      ← unit tests you write along the way
```

---

## Before You Start

### Understand these files first (read, don't modify)

| File | Why | Key things to note |
|---|---|---|
| `nanochat/gpt.py` (original, before SALA changes) | The base model you're extending | `GPTConfig`, `CausalSelfAttention.forward` signature `(x, ve, cos_sin, window_size, kv_cache)`, `Block`, `GPT.__init__`, `GPT.forward` loop, `setup_optimizers` |
| `nanochat/flash_attention.py` | How attention is dispatched | `flash_attn_func(q, k, v, causal, window_size)`, SDPA fallback, `(B,T,H,D)` tensor layout |
| `nanochat/engine.py` | KV cache for inference | `KVCache` class — shape `(n_layers, B, T, H, D)`, `advance()`, `get_layer_cache()` |
| `nanochat/muon.py` | Muon optimizer | **Only accepts 2D params** — this constraint shapes how you split optimizer groups |
| `nanochat/checkpoint_manager.py` | Save/load checkpoints | `build_model` flow: meta device → `to_empty` → `init_weights` → `load_state_dict`. Non-persistent buffers (like rotary embeddings) are NOT in the checkpoint. |
| `scripts/base_train.py` | Training loop pattern | Gradient accumulation, LR schedule, dual optimizer (Muon + AdamW), autocast, torch.compile |

### Papers to reference

| Paper | Key concepts to extract |
|---|---|
| `oldMoney-Project/SALA_paper.md` (MiniCPM-SALA section) | 1:3 ratio, HALO conversion, 5-stage pipeline, HyPE, output gates, layer selection |
| `oldMoney-Project/SALA_paper.md` (InfLLM-V2 section) | 3-stage block selection, shared KV params, dense-sparse switchable, `dense_len` threshold |
| `oldMoney-Project/SALA_paper.md` (Lightning Attention section) | Intra-block softmax + inter-block linear trick, chunk-based parallel training |

---

## Milestone 1: Lightning Attention Module

**Goal**: A standalone `LightningAttention` class with the exact same `forward` signature as `CausalSelfAttention`, so it can drop into `Block` unchanged.

### What to implement

```python
class LightningAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Full MHA: n_kv_head = n_head (not GQA)
        # Projections: c_q, c_k, c_v, c_proj (all Linear, no bias)
        # Learnable QK norm: nn.RMSNorm(head_dim) for q and k
        # Output norm: nn.RMSNorm(n_embd)
        # Output gate: z_proj = Linear(n_embd, n_head*head_dim) if config.use_output_gate
        # Value embedding gate: ve_gate (same as CausalSelfAttention)
        # ALiBi decay slopes: register_buffer("g_gamma", ..., persistent=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 1. Q, K, V projections → (B, T, H, D)
        # 2. Value embedding blend (copy logic from CausalSelfAttention)
        # 3. RoPE on Q, K (HyPE: linear layers GET RoPE)
        # 4. Learnable QK norm
        # 5. if training: chunk_simple_gla(q, k, v, g=log_slopes)
        #    if inference: fused_recurrent_simple_gla(q, k, v, g=log_slopes, initial_state=...)
        # 6. Output norm
        # 7. Output gate: y = y * sigmoid(z_proj(x))
        # 8. O projection
```

### Key decisions you need to make

1. **ALiBi slopes**: `2^(-8/n * i)` for `i=1..n_head`. Store as `log(slopes)` since `fla` expects log-space. These are non-learnable buffers.

2. **fla API**: Install `pip install fla`, then:
   ```python
   from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
   # chunk_simple_gla expects (B, H, T, D) — you need to transpose from (B, T, H, D)
   # g parameter shape: (B, H, T, 1) — broadcast from (H,) slopes
   ```

3. **Value embedding dimension**: Linear layers use full MHA, so VE dim = `n_head * head_dim` = `n_embd`. This differs from dense layers which may use `n_kv_head * head_dim`.

### How to verify

```python
# Create a tiny config and test forward/backward
config = GPTConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=256, sequence_len=128,
                   attn_types=["dense","linear","linear","dense"], use_output_gate=True)
layer = LightningAttention(config, layer_idx=1).cuda()
x = torch.randn(2, 128, 256).cuda().bfloat16()
cos_sin = (torch.randn(1,128,1,64).cuda().bfloat16(), torch.randn(1,128,1,64).cuda().bfloat16())
y = layer(x, ve=None, cos_sin=cos_sin, window_size=(-1,0), kv_cache=None)
assert y.shape == (2, 128, 256)
y.sum().backward()  # gradient flows?
print("Milestone 1 PASSED")
```

### Gotcha to watch for

- `g_gamma` is `persistent=False`. After checkpoint save/load, it will be garbage. You MUST recompute it in `init_weights()` — just like how rotary embeddings (`cos`, `sin`) are recomputed.

---

## Milestone 2: InfLLM-V2 Sparse Attention Wrapper

**Goal**: A Python module that wraps the InfLLM-V2 CUDA kernels for block-sparse attention.

### Setup

```bash
cd /Users/chen.zhe/code/oldMoney-Project/_official_pkgs/infllmv2_cuda_impl
pip install -e .
```

This builds the `infllm_v2` package with CUDA kernels. Verify: `python -c "from infllm_v2 import infllmv2_attn_varlen_func; print('OK')"`

### What to implement

```python
def compress_keys(k, kernel_size, stride):
    """Mean-pool keys along token dim. (total_k, H, D) → (compressed, H, D)"""

def sparse_block_selection(q, k, config, cu_seqlens_q, cu_seqlens_k, max_context_len):
    """3-stage pipeline:
    1. compress_keys at two granularities (k1: stride=16, k2: stride=64)
    2. infllmv2_attn_stage1(q, k1, k2) → attention scores
    3. max_pooling_1d_varlen(scores) → block-level scores
    4. topk selection → (n_kv_heads, total_q, topk) int32
    """

def sparse_attention_forward(q, k, v, topk_idx, ...):
    """Call infllmv2_attn_varlen_func with topk_idx mask."""
```

### Key facts about the CUDA kernel

- Block size is hardcoded at **64 tokens**
- `topk_idx` must be **int32**
- Stage1 kernel requires GQA ratio >= 16. If your model has fewer, **repeat query heads**: `q.repeat_interleave(16 // ratio, dim=1)`
- The kernel only compiles **bf16** variants for hdim 64 and 128

### How to verify

```python
# Test with dense attention and verify sparse gives similar output
# (with topk large enough to select all blocks, sparse ≈ dense)
```

### Can skip for now

This milestone is optional for initial training — you can start with Milestones 3-5 using only standard FlashAttention on sparse layers (`disable_sparse=True`). Come back to this when you reach Milestone 7 (long-context).

---

## Milestone 3: Hybrid GPT Model

**Goal**: Modify `GPT` so each layer can be either `CausalSelfAttention` or `LightningAttention`.

### What to change in `gpt.py` (copy the original to `03_hybrid_gpt.py` first)

**Copy from**: `nanochat/gpt.py` (the ORIGINAL before any SALA changes)

**Changes needed**:

1. **GPTConfig** — add these fields:
   ```
   attn_types: list = None       # None = all dense (backward compat)
   n_kv_head_sparse: int = -1    # KV heads for sparse layers
   use_output_gate: bool = False
   use_hype: bool = False
   rope_theta: float = 10000.0
   use_gradient_checkpointing: bool = False
   sparse_*: ...                 # InfLLM-V2 params (for later)
   ```

2. **CausalSelfAttention.__init__** — add:
   - `is_sparse_layer` param → controls `n_kv_head` (use `n_kv_head_sparse` if set)
   - `self.o_gate` (output gate, init zeros)
   - `self.use_rope` flag (HyPE: `False` for sparse layers)

3. **CausalSelfAttention.forward** — add:
   - Conditional RoPE: `if self.use_rope:`
   - Output gate before c_proj: `y = y * sigmoid(self.o_gate(x))`
   - **Remove** `kv_cache.advance(T)` from here (move to GPT.forward)

4. **Block.__init__** — accept `attn_type` arg:
   ```python
   if attn_type == "linear":
       self.attn = LightningAttention(config, layer_idx)
   else:
       self.attn = CausalSelfAttention(config, layer_idx, is_sparse_layer=True)
   ```

5. **GPT.__init__** — per-layer construction:
   ```python
   self.attn_types = config.attn_types or ["dense"] * config.n_layer
   self.disable_sparse = False
   blocks = [Block(config, i, attn_type=self.attn_types[i]) for i in range(n_layer)]
   ```
   - Value embed dimensions differ per layer type (linear=n_embd, sparse=n_kv_head_sparse*head_dim)
   - Use `config.rope_theta` instead of hardcoded 10000

6. **GPT.forward** — add:
   - Gradient checkpointing: `torch.utils.checkpoint.checkpoint(block, ..., use_reentrant=False)`
   - Move `kv_cache.advance(T)` to AFTER the layer loop (so it works for both layer types)

7. **GPT.setup_optimizers** — Muon only gets 2D params:
   ```python
   for p in self.transformer.h.parameters():
       if p.ndim == 2: matrix_params.append(p)    # → Muon
       else: scalar_params.append(p)              # → AdamW
   ```

8. **GPT.init_weights** — add init for:
   - `o_gate`, `z_proj`: zeros
   - `q_norm`, `k_norm`, `o_norm`: nn.RMSNorm defaults (ones)
   - Recompute `g_gamma` for LightningAttention layers (non-persistent buffer)

### How to verify

```python
# 1. Build a dense-only model (attn_types=None) and verify it still works identically
# 2. Build a hybrid model and verify forward/backward pass
# 3. Check param count: hybrid should have ~3% more params than dense (MHA on linear layers)
config_dense = GPTConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=256)
config_hybrid = GPTConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=256,
                          attn_types=["dense","linear","linear","dense"],
                          use_output_gate=True, use_hype=True)
# Both should produce (B, T, vocab_size) logits and train
```

---

## Milestone 4: HALO Conversion (Dense → Hybrid)

**Goal**: Load a trained dense checkpoint, create a hybrid model, copy weights.

### The algorithm

```
1. Load dense model checkpoint (from base_train)
2. Choose which layers stay sparse: [0, 4, 9, 13, 18, 22, 27, 31] (uniform spacing)
   - First/last layer always sparse
   - Or implement actual HALO selection (see below)
3. Build hybrid model with new config (attn_types set)
4. For each layer:
   - Sparse layer: copy Q/K/V/O directly. If n_kv_head changes, compress by averaging groups.
   - Linear layer: copy Q/O directly. Expand K/V by repeating (GQA→MHA).
5. Copy all non-layer params: wte, lm_head, resid_lambdas, x0_lambdas, value_embeds
6. Init new params to zero: o_gate, z_proj
7. Save hybrid checkpoint
```

### How to copy from `nanochat/checkpoint_manager.py`

Use `build_model()` to load the dense checkpoint. This handles:
- `_patch_missing_config_keys` (adds defaults for new fields)
- Meta device init → `to_empty` → `init_weights` → `load_state_dict`

### The tricky part: GQA → MHA expansion for K/V

```python
# Dense model: K weight shape = (n_kv_head * head_dim, n_embd) e.g. (256, 2048)
# Hybrid linear layer: K weight = (n_head * head_dim, n_embd) e.g. (2048, 2048)
# Expand by repeating each KV head group:
k_w = dense_k.weight.view(n_kv_head, head_dim, n_embd)  # (2, 128, 2048)
k_expanded = k_w.repeat(n_head // n_kv_head, 1, 1)       # (16, 128, 2048)
hybrid_k.weight.data.copy_(k_expanded.view(-1, n_embd))   # (2048, 2048)
```

### Optional: Real HALO layer selection

Instead of uniform spacing, measure which layers matter most:
```
for each layer i:
    temporarily convert layer i to linear attention
    measure perplexity on validation set
    restore layer i
keep the 8 layers with the highest perplexity increase as sparse
```

### How to verify

```python
# Load dense checkpoint, convert, check:
# 1. No NaN/Inf in logits
# 2. Loss is finite (will be higher than dense, but not catastrophic)
# 3. Weight shapes match (print all param shapes)
# 4. Compare logits between dense model and hybrid model's sparse layers — they should be identical
```

---

## Milestone 5: HALO Fine-tune (Train Linear Layers Only)

**Goal**: Train only the newly converted linear attention layers while freezing everything else.

### What to copy from

Start from `scripts/base_train.py` and simplify:

### Key differences from base_train

1. **Load hybrid checkpoint** (from Milestone 4)
2. **Freeze everything except LightningAttention**:
   ```python
   for name, param in model.named_parameters():
       param.requires_grad = False
   for block in model.transformer.h:
       if isinstance(block.attn, LightningAttention):
           for param in block.attn.parameters():
               param.requires_grad = True
   ```
3. **Optimizer**: AdamW only (no Muon needed). Use DistAdamW if DDP.
4. **Short training**: ~1.3B tokens, context length 512, LR 7.5e-3, 2000-step warmup then constant
5. **`model.disable_sparse = True`** — sparse layers use standard FlashAttention

### How to verify

```python
# 1. Only linear attention params have requires_grad=True
# 2. Loss decreases over training (starts high from random linear attn, converges toward dense level)
# 3. Frozen params haven't changed (compare state_dict before/after)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {trainable/total*100:.1f}% of params")  # should be ~20-30%
```

---

## Milestone 6: Continual Stable-Training

**Goal**: Unfreeze all parameters and train the full hybrid model at 4K context, sparse disabled.

### Key differences from Milestone 5

1. **All params unfrozen**: `param.requires_grad = True` for everything
2. **Full optimizer**: `model.setup_optimizers()` — Muon for 2D, AdamW for 1D/embeddings
3. **`model.disable_sparse = True`** still — sparse layers use dense FlashAttention with sliding window
4. **Longer training**: ~30B tokens at 4K context
5. **LR**: 7.5e-3 with 2000-step warmup, then constant

### What to watch for

- Loss should NOT spike when unfreezing — if it does, reduce LR
- Linear layers and sparse layers should both receive gradients (check with `param.grad is not None`)
- This is the longest stage besides pretraining

### How to verify

```python
# 1. Loss is smooth (no spikes at start)
# 2. All param groups have non-zero gradients
# 3. Eval perplexity approaches dense model's level
```

---

## Milestone 7: Long-Context Adaptation

**Goal**: Enable InfLLM-V2 sparse attention and progressively extend context to 128K.

### Key changes

1. **`model.disable_sparse = False`** — sparse layers now use InfLLM-V2 block selection for seq > `sparse_dense_len` (8192)
2. **Progressive extension**: 4K → 32K → 64K → 128K (separate training phases)
3. **`rope_theta = 100000`** — NTK-aware scaling for long context
4. **`use_gradient_checkpointing = True`** — essential to fit 128K in memory
5. **Reduce device_batch_size** as context grows (8 → 2 → 1)

### Memory budget at 128K (d32, batch=1, 8xH100)

```
Model params (bf16):     ~4.4 GB
Optimizer states:        ~13 GB
Activations (w/ ckpt):   ~1 GB  (vs ~16 GB without checkpointing)
KV cache (8 sparse):     ~1 GB
Linear states (24 layers): ~10 MB
─────────────────────────────────
Total per GPU:           ~20 GB  ← fits in 80 GB H100
```

### Training phases

| Phase | Context | batch | LR | Tokens | Time est. |
|---|---|---|---|---|---|
| 4a | 32K | 2 | 3e-4 → 2e-4 | 10B | ~15h |
| 4b | 64K | 1 | 2e-4 → 1e-4 | 6B | ~15h |
| 4c | 128K | 1 | 1e-4 → 3.75e-5 | 5B | ~20h |

### How to verify

```python
# 1. No OOM at 32K with batch=2 and gradient checkpointing
# 2. InfLLM-V2 sparse dispatch is actually invoked (log topk_idx shape per step)
# 3. Perplexity on long sequences improves across phases
# 4. Short-sequence performance doesn't regress (eval on standard benchmarks)
```

---

## Appendix: HybridKVCache for Inference

When you reach inference with the hybrid model, you'll need a modified KV cache:

```python
class HybridKVCache(KVCache):
    # Sparse layers: standard KV cache (grows with context)
    # Linear layers: fixed recurrent state (B, H, D, D) — constant size
    # Key bug to avoid: n_layers must equal TOTAL layer count, not just sparse count
    #   (otherwise kv_cache.advance() never triggers)
```

---

## Appendix: Things That Will Bite You

1. **Muon rejects 1D params** — if you put RMSNorm weights or scalar params into Muon, it crashes with an assertion error. Filter by `p.ndim == 2`.

2. **g_gamma disappears after save/load** — it's `persistent=False`, so it's not saved to checkpoints. You must recompute it in `init_weights()`.

3. **kv_cache.advance() placement** — if advance() is inside CausalSelfAttention and the last layer is linear, advance never fires. Put it in GPT.forward after the layer loop.

4. **fla tensor layout** — fla expects `(B, H, T, D)`, nanochat uses `(B, T, H, D)`. Transpose before/after calling fla kernels.

5. **GQA→MHA value embedding expansion** — linear layers have `n_head` KV heads but the dense model's value embeddings have `n_kv_head` entries. You need to repeat-expand them during HALO conversion.

6. **autocast on non-CUDA** — don't use `torch.no_grad()` as a fallback for `torch.amp.autocast`. Use `contextlib.nullcontext()`. Otherwise backward() breaks.

7. **DDP gradient sync** — if you use plain `torch.optim.AdamW` without DDP wrapping, each GPU trains independently. Use `DistAdamW` or wrap the model in `DistributedDataParallel`.

---

## Progress Tracker

- [ ] Milestone 1: LightningAttention module
- [ ] Milestone 2: InfLLM-V2 sparse wrapper
- [ ] Milestone 3: Hybrid GPT model
- [ ] Milestone 4: HALO conversion script
- [ ] Milestone 5: HALO fine-tune
- [ ] Milestone 6: Continual stable-training
- [ ] Milestone 7: Long-context adaptation
- [ ] Bonus: HybridKVCache + inference
- [ ] Bonus: Real HALO layer selection (data-driven)
