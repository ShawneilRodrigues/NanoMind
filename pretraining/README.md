# NanoMind Pretraining

This directory contains the complete pretraining setup for NanoMind, featuring:

- **fp16 precision** - Half precision training for efficiency
- **Flash Attention** - Optimized attention implementation
- **torch.compile** - Graph compilation for maximum speed
- **Fused AdamW** - Optimized optimizer
- **2× T4 DDP** - Distributed Data Parallel on multiple GPUs
- **OpenWebText** - Large-scale text dataset
- **WandB integration** - Experiment tracking and visualization
- **HuggingFace checkpointing** - Automatic checkpoint management

## File Structure

```
pretraining/
├── model.py           # NanoMind architecture with GQA, MoE support
├── train.py           # Distributed training script with streaming dataset
├── run_ddp.sh         # Launch script for 2-GPU DDP training
├── checkpoints/       # Local checkpoint storage
└── README.md          # This file
```

## Quick Start

### 1. Set Environment Variables

```bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"
export HF_REPO_ID="your_username/NanoMind"
```

### 2. Launch Training

```bash
bash run_ddp.sh
```

Or with custom parameters:

```bash
torchrun --nproc_per_node=2 --master_port=29501 train.py \
    --max_steps 500000 \
    --batch_size 4 \
    --accum_steps 8 \
    --compile \
    --lr 3e-4
```

## Key Features

### Model Config

- **d_model**: 512 (embedding dimension)
- **n_heads**: 8 (attention heads)
- **n_kv_heads**: 2 (key-value heads for GQA)
- **n_layers**: 8 (transformer blocks)
- **max_seq_len**: 512 (sequence length)
- **use_moe**: false (enable Mixture of Experts)

### Training Config

- **Dataset**: OpenWebText (streaming)
- **Optimizer**: AdamW (fused)
- **LR Schedule**: Cosine annealing with warmup
- **Gradient Accumulation**: 8 steps
- **Grad Clip**: 1.0
- **Save Every**: 1500 steps
- **Sample Every**: 1000 steps

### Checkpoint Management

- **Local**: Keeps latest 5 checkpoints
- **HuggingFace**: Keeps latest 2 checkpoints
- **Auto-resume**: Detects and resumes from latest checkpoint

## Resume Training

To resume from a specific checkpoint:

```bash
torchrun --nproc_per_node=2 train.py \
    --resume_from ./checkpoints/checkpoint_step_XXXXX.pt \
    ...other args
```

Or recover from HuggingFace if session expires:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="shawneil/NanoMind", 
                filename="checkpoint_step_XXXXX.pt",
                local_dir="./checkpoints")
```

## Monitoring

- **Training logs**: Check `train.log` for real-time progress
- **WandB**: View metrics, samples, and model status at https://wandb.ai
- **Checkpoints**: Stored locally in `checkpoints/` and synced to HuggingFace

## Requirements

- PyTorch with CUDA support
- 2× GPU (tested on T4s, adaptable to any GPU)
- Dependencies: wandb, bitsandbytes, datasets, tiktoken, transformers

See `run_ddp.sh` for automatic dependency installation.
