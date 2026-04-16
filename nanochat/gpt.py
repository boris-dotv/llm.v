"""
GPT model — MiniCPM-SALA aligned architecture.
Notable features:
- Rotary embeddings (RoPE)
- QK norm (parameter-free on softmax heads, learnable on lightning heads)
- SwiGLU MLP (gate_proj, up_proj, down_proj)
- Learnable RMSNorm (input_layernorm, post_attention_layernorm, final ln_f)
- MiniCPM scaling: scale_emb on embedding, scale_depth residual scaling, scale_width on logits
- Tied or untied word embeddings (tie_word_embeddings flag)
- Hybrid mixer_types: minicpm4 (softmax), lightning (SimpleGLA), sparse (InfLLM-V2)
- GQA support, output gates, HyPE
- No bias in linear layers
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    intermediate_size: int = -1  # MLP intermediate size. -1 = use 4*n_embd (legacy default).
    scale_depth: float = 1.4  # Residual scaling: sublayer_output * (scale_depth / sqrt(n_layer))
    rms_norm_eps: float = 1e-6  # Epsilon for RMSNorm
    scale_emb: float = 8.0  # Multiply embedding output by this factor
    dim_model_base: int = 256  # Base model dim for logit width scaling
    tie_word_embeddings: bool = False  # Share embed_tokens and lm_head weights
    # --- SALA hybrid attention fields ---
    # Per-layer mixer type: "minicpm4" (dense softmax), "lightning" (SimpleGLA), "sparse" (InfLLM-V2)
    mixer_types: list = None  # None = all minicpm4
    # KV heads for dense/sparse layers. -1 = use n_kv_head. Set to e.g. 2 for GQA on sparse layers.
    n_kv_head_sparse: int = -1
    # Add sigmoid output gate after attention (o_gate on sparse layers, z_proj on linear layers)
    use_output_gate: bool = False
    # HyPE: apply RoPE on linear layers, NoPE on sparse layers (for long-range recall)
    use_hype: bool = False
    # RoPE base frequency. Increase for long context (e.g. 100000 for NTK-aware scaling).
    rope_theta: float = 10000.0
    # Wrap each block forward in torch.utils.checkpoint for memory-efficient long-context training
    use_gradient_checkpointing: bool = False
    # InfLLM-V2 sparse attention config (only used when sparse is enabled on dense-type layers)
    sparse_block_size: int = 64
    sparse_kernel_size: int = 32      # k1 compression kernel window
    sparse_kernel_stride: int = 16    # k1 compression stride
    sparse_topk: int = 63             # top-k blocks to select
    sparse_init_blocks: int = 1       # initial blocks always attended
    sparse_local_blocks: int = 32     # local/sliding window blocks
    sparse_dense_len: int = 8192      # below this seq length, sparse layers use standard dense attn


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, is_sparse_layer=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        # Sparse layers can use fewer KV heads (GQA) for KV compression
        if is_sparse_layer and config.n_kv_head_sparse > 0:
            self.n_kv_head = config.n_kv_head_sparse
        else:
            self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # Output gate (SALA: o_gate on sparse layers)
        self.o_gate = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False) if config.use_output_gate else None
        # HyPE: sparse layers do NOT use RoPE (NoPE for long-range recall)
        self.use_rope = not (config.use_hype and is_sparse_layer)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings (conditionally — HyPE: sparse layers skip RoPE)
        if self.use_rope:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)

        # Output gate (SALA): sigmoid gating before projection
        if self.o_gate is not None:
            y = y * torch.sigmoid(self.o_gate(x))

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate = config.intermediate_size if config.intermediate_size > 0 else 4 * config.n_embd
        self.gate_proj = nn.Linear(config.n_embd, intermediate, bias=False)
        self.up_proj = nn.Linear(config.n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx, mixer_type="minicpm4"):
        super().__init__()
        self.residual_scale = config.scale_depth / (config.n_layer ** 0.5)
        self.input_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        if mixer_type == "lightning":
            from nanochat.linear_attention import LightningAttention
            self.attn = LightningAttention(config, layer_idx)
        else:
            # "minicpm4" or "sparse" — both use softmax attention
            is_sparse = mixer_type == "sparse"
            self.attn = CausalSelfAttention(config, layer_idx, is_sparse_layer=is_sparse)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(self.input_layernorm(x), cos_sin, window_size, kv_cache) * self.residual_scale
        x = x + self.mlp(self.post_attention_layernorm(x)) * self.residual_scale
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Resolve per-layer mixer types (None = all minicpm4)
        self.mixer_types = config.mixer_types or ["minicpm4"] * config.n_layer
        assert len(self.mixer_types) == config.n_layer, f"mixer_types length {len(self.mixer_types)} != n_layer {config.n_layer}"
        # Flag to disable InfLLM-V2 sparse dispatch (sparse layers fall back to dense FlashAttention).
        # Lightning layers are NOT affected — they always use SimpleGLA.
        self.disable_sparse = False
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Construct blocks with per-layer mixer type
        blocks = []
        for layer_idx in range(config.n_layer):
            mixer_type = self.mixer_types[layer_idx]
            blocks.append(Block(config, layer_idx, mixer_type=mixer_type))
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),
            "ln_f": nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        })
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
          CausalSelfAttention (dense/sparse layers):
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            attn.o_gate:     zeros (if present)
          LightningAttention (linear layers):
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            attn.z_proj:     zeros (if present)
            attn.q_norm/k_norm/o_norm: ones (standard RMSNorm init, handled by nn.RMSNorm default)
          MLP:
            mlp.gate_proj:   uniform, std=1/sqrt(n_embd)
            mlp.up_proj:     uniform, std=1/sqrt(n_embd)
            mlp.down_proj:   zeros
        """
        from nanochat.linear_attention import LightningAttention

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if self.lm_head is not None:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            attn = block.attn
            torch.nn.init.uniform_(attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(attn.c_proj.weight)
            # Output gates (o_gate for dense, z_proj for linear): init to zero
            if hasattr(attn, 'o_gate') and attn.o_gate is not None:
                torch.nn.init.zeros_(attn.o_gate.weight)
            if hasattr(attn, 'z_proj') and attn.z_proj is not None:
                torch.nn.init.zeros_(attn.z_proj.weight)
            # MLP (SwiGLU)
            torch.nn.init.uniform_(block.mlp.gate_proj.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.up_proj.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.down_proj.weight)
            # Learnable RMSNorm weights (meta device + to_empty leaves these as garbage)
            block.input_layernorm.weight.fill_(1.0)
            block.post_attention_layernorm.weight.fill_(1.0)
        self.transformer.ln_f.weight.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Recompute non-persistent buffers for LightningAttention layers (ALiBi decay slopes)
        from nanochat.linear_attention import LightningAttention, _build_slope_tensor
        for block in self.transformer.h:
            if isinstance(block.attn, LightningAttention):
                device = block.attn.c_q.weight.device
                slopes = _build_slope_tensor(block.attn.n_head).to(device)
                block.attn.g_gamma = torch.log(slopes)

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        if base is None:
            base = self.config.rope_theta
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, attention FLOPs vary per layer type:
        - Dense/sparse layers: 12 * h * q * effective_seq_len (softmax attention)
        - Linear layers: 2 * T * n_head * head_dim^2 (recurrent state update, no quadratic attention)
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings
        nparams_exclude = self.transformer.wte.weight.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for layer type
        attn_flops = 0
        for i in range(self.config.n_layer):
            if self.mixer_types[i] == "lightning":
                # Linear attention: state update is O(T * n_head * head_dim^2), no quadratic attention
                attn_flops += 6 * t * h * q * q  # 6x for fwd+bwd of the state matmul
            else:
                # minicpm4 or sparse: full context softmax attention
                attn_flops += 12 * h * q * t
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        # Muon requires strictly 2D parameters, so we split transformer.h params into 2D (Muon) and non-2D (AdamW)
        matrix_params = []       # 2D weight matrices -> Muon
        attn_scalar_params = []  # 1D params from attention (norms, etc.) -> AdamW
        for p in self.transformer.h.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                attn_scalar_params.append(p)
        embedding_params = list(self.transformer.wte.parameters())
        ln_f_params = list(self.transformer.ln_f.parameters())
        lm_head_params = list(self.lm_head.parameters()) if self.lm_head is not None else []
        all_params = len(matrix_params) + len(attn_scalar_params) + len(embedding_params) + len(ln_f_params) + len(lm_head_params)
        assert len(list(self.parameters())) == all_params, f"Parameter count mismatch: {len(list(self.parameters()))} != {all_params}"
        # Create the AdamW optimizer for the embedding, lm_head, and 1D attention params
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        if lm_head_params:
            adam_groups.insert(0, dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale))
        # Add 1D attention params (norms, gates) to AdamW if any exist (SALA linear layers have these)
        # Also includes per-block input_layernorm + post_attention_layernorm weights
        all_scalar_params = attn_scalar_params + ln_f_params
        if all_scalar_params:
            adam_groups.append(dict(params=all_scalar_params, lr=scalar_lr))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the 2D linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) * self.config.scale_emb
        # Full causal attention window: (-1, 0) means attend to all previous tokens
        window_size = (-1, 0)
        for i, block in enumerate(self.transformer.h):
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos_sin, window_size, kv_cache,
                    use_reentrant=False,
                )
            else:
                x = block(x, cos_sin, window_size, kv_cache)
        # Advance KV cache position after all layers have processed (inference only)
        if kv_cache is not None:
            kv_cache.advance(T)
        x = self.transformer.ln_f(x)

        # Forward the lm_head (compute logits)
        scale_width = self.config.n_embd / self.config.dim_model_base
        if self.lm_head is not None:
            logits = self.lm_head(x / scale_width)
        else:
            logits = F.linear(x / scale_width, self.transformer.wte.weight)
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for loss computation
        # softcap = 15  # logit softcap removed — MiniCPM doesn't use it
        # logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
