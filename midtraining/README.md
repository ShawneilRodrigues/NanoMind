# NanoMind SFT (Supervised Fine-Tuning)

Supervised Fine-Tuning on Alpaca dataset with the NanoMind model.

## Features

- **Loss ONLY on response tokens** - Instruction/input tokens are masked during loss computation
- **fp16 precision** - Half precision training for efficiency
- **Flash Attention** - Optimized attention implementation
- **torch.compile** - Graph compilation for maximum speed
- **Fused AdamW** - Optimized optimizer
- **2× T4 DDP** - Distributed Data Parallel on multiple GPUs
- **Alpaca dataset** - 52K instruction-following examples
- **WandB integration** - Experiment tracking and visualization
- **HuggingFace checkpointing** - Automatic checkpoint management

## File Structure

```
midtraining/
├── model.py           # NanoMind architecture with loss_mask support
├── sft_data.py        # Alpaca dataset loader with loss masking
├── sft_train.py       # SFT training script with DDP
├── run_sft.sh         # Launch script for 2-GPU DDP training
├── sft_checkpoints/   # Local checkpoint storage
└── README.md          # This file
```

## Quick Start

### 1. Set Environment Variables

```bash
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_huggingface_token"
export HF_REPO_ID="your_username/NanoMind-SFT"
```

### 2. Launch Training

```bash
bash run_sft.sh
```

Or with custom parameters:

```bash
torchrun --nproc_per_node=2 --master_port=29502 sft_train.py \
    --pretrain_repo "shawneil/NanoMind" \
    --max_epochs 5 \
    --batch_size 8 \
    --accum_steps 4 \
    --compile \
    --lr 2e-5
```

## Key Features

### Model Config

- **d_model**: 512 (embedding dimension)
- **n_heads**: 8 (attention heads)
- **n_kv_heads**: 2 (key-value heads for GQA)
- **n_layers**: 8 (transformer blocks)
- **max_seq_len**: 512 (sequence length)

### Training Config

- **Dataset**: Alpaca (52K examples)
- **Optimizer**: AdamW (fused)
- **LR Schedule**: Cosine annealing with warmup
- **Learning Rate**: 2e-5 (lower than pretraining for fine-tuning)
- **Gradient Accumulation**: 4 steps
- **Batch Size**: 8 per GPU
- **Grad Clip**: 1.0
- **Max Epochs**: 5
- **Warmup Steps**: 100
- **Save Every**: 500 steps

### Loss Masking

The model supports `loss_mask` which enables computing loss ONLY on response tokens:

```python
logits, loss = model(input_ids, targets, loss_mask=mask)
# mask[i] = 0 for instruction/input tokens
# mask[i] = 1 for response tokens
```

### Checkpoint Management

- **Local**: Keeps latest 3 checkpoints
- **HuggingFace**: Keeps latest 2 checkpoints
- **Auto-resume**: Detects and resumes from latest SFT checkpoint
- **Pretrained loading**: Auto-downloads from HF if no local pretraining checkpoint

## Resume Training

### Continue SFT from checkpoint:
```bash
torchrun --nproc_per_node=2 sft_train.py \
    --out_dir ./sft_checkpoints \
    ...other args
```
Auto-detection will resume from the latest SFT checkpoint.

### Start SFT from custom pretrained:
```bash
torchrun --nproc_per_node=2 sft_train.py \
    --pretrain_ckpt /path/to/checkpoint_step_5000.pt \
    ...other args
```

## Dataset Format

The Alpaca dataset is formatted as:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

Loss is only computed on the `Response:` portion, allowing the model to learn from instruction-following examples without penalizing for formatting.

## Monitoring

- **Training logs**: Check `sft_train.log` for real-time progress
- **WandB**: View metrics, samples, and model status at https://wandb.ai
- **Checkpoints**: Stored locally in `sft_checkpoints/` and synced to HuggingFace

## Requirements

- PyTorch with CUDA support
- 2× GPU (tested on T4s, adaptable to any GPU)
- Dependencies: wandb, datasets, tiktoken, transformers

See `run_sft.sh` for automatic dependency installation.

## Architecture Details

### NanoMind with SFT Support

The NanoMind model extended for SFT includes:
- **loss_mask parameter**: Enables selective loss computation
- **Grouped Query Attention (GQA)**: Efficient attention with key/value heads shared
- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Positional Embeddings
- **SwiGLU**: Gated Linear Unit activation
- **Optional MoE**: Mixture of Experts support

## References

- Alpaca Dataset: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- Architecture: Grouped Query Attention, RoPE, SwiGLU
- Training: DDP with gradient accumulation and mixed precision
