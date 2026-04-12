"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
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
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # --- SALA hybrid attention fields ---
    # Per-layer attention type: "dense" (softmax, optionally sparse via InfLLM-V2) or "linear"
    # (SimpleGLA). None = all dense (backward compat with existing checkpoints).
    attn_types: list = None
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


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

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
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Output gate (SALA: o_gate on sparse layers)
        self.o_gate = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False) if config.use_output_gate else None
        # HyPE: sparse layers do NOT use RoPE (NoPE for long-range recall)
        self.use_rope = not (config.use_hype and is_sparse_layer)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, attn_type="dense"):
        super().__init__()
        if attn_type == "linear":
            from nanochat.linear_attention import LightningAttention
            self.attn = LightningAttention(config, layer_idx)
        else:
            # "dense" — used for both standard dense and InfLLM-V2 sparse attention
            is_sparse = attn_type == "dense" and config.attn_types is not None
            self.attn = CausalSelfAttention(config, layer_idx, is_sparse_layer=is_sparse)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
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
        # Resolve per-layer attention types (None = all dense for backward compat)
        self.attn_types = config.attn_types or ["dense"] * config.n_layer
        assert len(self.attn_types) == config.n_layer, f"attn_types length {len(self.attn_types)} != n_layer {config.n_layer}"
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # For linear layers, window_size is ignored — set to (-1, 0) to avoid confusion
        for i in range(config.n_layer):
            if self.attn_types[i] == "linear":
                self.window_sizes[i] = (-1, 0)
        # Flag to disable InfLLM-V2 sparse dispatch (sparse layers fall back to dense FlashAttention).
        # Linear layers are NOT affected — they always use SimpleGLA.
        self.disable_sparse = False
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Construct blocks with per-layer attention type
        blocks = []
        for layer_idx in range(config.n_layer):
            attn_type = self.attn_types[layer_idx]
            blocks.append(Block(config, layer_idx, attn_type=attn_type))
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        # For linear layers (full MHA), kv_dim = n_head * head_dim = n_embd
        # For dense/sparse layers, kv_dim = n_kv_head * head_dim (may be smaller with GQA)
        head_dim = config.n_embd // config.n_head
        ve_dims = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                if self.attn_types[i] == "linear":
                    ve_dims[str(i)] = config.n_head * head_dim  # full MHA
                else:
                    kv_heads = config.n_kv_head_sparse if config.n_kv_head_sparse > 0 else config.n_kv_head
                    ve_dims[str(i)] = kv_heads * head_dim
        self.value_embeds = nn.ModuleDict({k: nn.Embedding(padded_vocab_size, dim) for k, dim in ve_dims.items()})
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
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """
        from nanochat.linear_attention import LightningAttention

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
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
            # MLP
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

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
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

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

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

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
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for layer type and sliding window
        attn_flops = 0
        for i, window_size in enumerate(self.window_sizes):
            if self.attn_types[i] == "linear":
                # Linear attention: state update is O(T * n_head * head_dim^2), no quadratic attention
                attn_flops += 6 * t * h * q * q  # 6x for fwd+bwd of the state matmul
            else:
                window = window_size[0]  # (left, right) tuple, we use left
                effective_seq = t if window < 0 else min(window, t)
                attn_flops += 12 * h * q * effective_seq
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
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        all_params = len(matrix_params) + len(attn_scalar_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        assert len(list(self.parameters())) == all_params, f"Parameter count mismatch: {len(list(self.parameters()))} != {all_params}"
        # Create the AdamW optimizer for the embedding, lm_head, per-layer scalars, and 1D attention params
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale),  # same LR as token embedding
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        # Add 1D attention params (norms, gates) to AdamW if any exist (SALA linear layers have these)
        if attn_scalar_params:
            adam_groups.append(dict(params=attn_scalar_params, lr=scalar_lr))
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
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, ve, cos_sin, self.window_sizes[i], kv_cache,
                    use_reentrant=False,
                )
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        # Advance KV cache position after all layers have processed (inference only)
        if kv_cache is not None:
            kv_cache.advance(T)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

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
