#!/bin/bash
set -e

# Configuration
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-shawneil/NanoMind-SFT}"
PRETRAIN_REPO="shawneil/NanoMind"
SAVE_EVERY=500
MAX_EPOCHS=5
KEEP_LOCAL_N=3
KEEP_HF_N=2

# Setup
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"
export TORCHINDUCTOR_CACHE_DIR="./torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Install dependencies
python -m pip install -q \
    wandb datasets huggingface_hub \
    tiktoken transformers

# Launch training
torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    sft_train.py \
    --pretrain_repo    "$PRETRAIN_REPO" \
    --max_seq_len      512 \
    --batch_size       8 \
    --accum_steps      4 \
    --lr               2e-5 \
    --max_epochs       $MAX_EPOCHS \
    --warmup_steps     100 \
    --compile \
    --keep_local_n     $KEEP_LOCAL_N \
    --keep_hf_n        $KEEP_HF_N \
    --save_every       $SAVE_EVERY \
    --log_every        10 \
    --hf_token         "$HF_TOKEN" \
    --hf_repo_id       "$HF_REPO_ID" \
    --wandb_project    nano-nanomind-sft \
    --wandb_api_key    "$WANDB_API_KEY" \
    --out_dir          ./sft_checkpoints \
    2>&1 | tee sft_train.log

echo "✅ SFT Training Complete"
