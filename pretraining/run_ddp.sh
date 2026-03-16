#!/bin/bash
set -e

# Configuration
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-shawneil/NanoMind}"
SAMPLE_PROMPT="Once upon a time in a land far away"
SAVE_EVERY=1500
SAMPLE_EVERY=1000
MAX_STEPS=500000
KEEP_LOCAL_N=5
KEEP_HF_N=2

# Setup
export WANDB_API_KEY="$WANDB_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export HF_REPO_ID="$HF_REPO_ID"
export TORCHINDUCTOR_CACHE_DIR="./torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Install dependencies
python -m pip install -q \
    wandb bitsandbytes datasets huggingface_hub \
    tiktoken transformers ninja

# Launch training
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    train.py \
    --dataset          "Skylion007/openwebtext" \
    --text_field       text \
    --max_seq_len      512 \
    --batch_size       4 \
    --accum_steps      8 \
    --compile \
    --lr               3e-4 \
    --max_steps        $MAX_STEPS \
    --warmup_steps     500 \
    --keep_local_n     $KEEP_LOCAL_N \
    --keep_hf_n        $KEEP_HF_N \
    --save_every       $SAVE_EVERY \
    --sample_every     $SAMPLE_EVERY \
    --sample_prompt    "$SAMPLE_PROMPT" \
    --sample_max_new   200 \
    --hf_token         "$HF_TOKEN" \
    --hf_repo_id       "$HF_REPO_ID" \
    --wandb_project    nano-nanomind \
    --wandb_api_key    "$WANDB_API_KEY" \
    --out_dir          ./checkpoints \
    2>&1 | tee train.log
