"""
608M dense dry-run model config for S0 pretraining.

This config is a scaled-down version of the SALA 9B architecture, designed to
validate the full S0 training pipeline before scaling to 3.5B.

SALA-ALIGNED properties (carry over to dryrun_3p5b.py unchanged):
  - depth=32               (same as SALA 9B)
  - GQA 16:1               (n_head=16, n_kv_head=1, strict SALA ratio)
  - scale_emb=12.0         (muP value from MiniCPM-SALA/MiniCPM-4)
  - scale_depth=1.4        (MiniCPM residual scaling convention)
  - dim_model_base=256     (MiniCPM logit width scaling base)
  - intermediate=4*hidden  (SALA 9B uses 16384 = 4*4096; we use 4096 = 4*1024)
  - tie_word_embeddings=False
  - all minicpm4 mixer     (dense softmax attention + RoPE, every layer)
  - RMSNorm eps=1e-6
  - SwiGLU activation      (gate_proj * up_proj with SiLU gating)

DEVIATED properties (revisit when building dryrun_3p5b.py):
  - head_dim=64            (SALA 9B uses 128; unavoidable at this model size
                            with depth=32. head_dim is a kernel-level choice,
                            not architectural — FlashAttention handles both.)
  - vocab_size             (default 65536; overridden to 151936 when placeholder
                            Qwen2.5 tokenizer is active. The real bilingual
                            tokenizer from Track B will set the final value.)
"""


def get_model_config(vocab_size=65536):
    """Return GPTConfig kwargs for the 608M dense dry-run model.

    Args:
        vocab_size: Vocabulary size. Defaults to 65536 (design-time estimate).
                    Pass the actual tokenizer's vocab_size at runtime.

    Param count (approx):
        vocab=65536  -> 608M total params
        vocab=151936 -> 785M total params (Qwen2.5 placeholder tokenizer)
    """
    return dict(
        n_embd=1024,
        n_layer=32,
        n_head=16,
        n_kv_head=1,
        intermediate_size=4096,  # 4 * n_embd (strict SALA ratio)
        vocab_size=vocab_size,
        sequence_len=2048,       # dry-run default; 4096 for 3.5B
        mixer_types=["minicpm4"] * 32,
        tie_word_embeddings=False,
        scale_emb=12.0,          # muP (MiniCPM-SALA 9B / MiniCPM-4 value)
        scale_depth=1.4,
        dim_model_base=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )
