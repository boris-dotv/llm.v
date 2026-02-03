## Chat SFT
timestamp: 2026-02-03 12:18:55

- run: d26_sft_01
- device_type: 
- dtype: bfloat16
- source: mid
- model_tag: d26
- model_step: 818
- num_epochs: 1
- num_iterations: -1
- device_batch_size: 4
- target_examples_per_step: 32
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 22,443
- Number of iterations: 701
- Training loss: 0.3887
- Validation loss: 0.8002

