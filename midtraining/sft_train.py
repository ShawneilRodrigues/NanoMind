import os, sys, math, time, json, shutil, argparse
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True

@contextmanager
def nullctx():
    yield

def init_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()

def cleanup():
    dist.destroy_process_group()

def cosine_lr(step, warmup, total, base, min_lr=1e-5):
    if step < warmup:
        return base * step / max(1, warmup)
    if step >= total:
        return min_lr
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))

# ── CheckpointManager ─────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, out_dir, keep_local_n, keep_hf_n, hf_repo, hf_token):
        self.out_dir      = out_dir
        self.keep_local_n = keep_local_n
        self.keep_hf_n    = keep_hf_n
        self.hf_repo      = hf_repo
        self.hf_token     = hf_token
        self._local = sorted(
            [str(p) for p in Path(out_dir).glob("sft_step_*.pt")],
            key=lambda x: int(Path(x).stem.split("_")[-1])
        )

    def save(self, model, optimizer, step, epoch):
        raw = model.module
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
        new_path = os.path.join(self.out_dir, f"sft_step_{step}.pt")

        # disk guard
        free_mb = shutil.disk_usage(self.out_dir).free / 1024 / 1024
        if free_mb < 400 and self._local:
            oldest = self._local.pop(0)
            if os.path.isfile(oldest): os.remove(oldest)

        torch.save({
            "step": step, "epoch": epoch,
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(raw.cfg),
        }, new_path)
        size_mb = os.path.getsize(new_path) / 1024 / 1024
        print(f"[CKPT] Saved {Path(new_path).name} ({size_mb:.1f} MB)", flush=True)
        self._local.append(new_path)

        self._hf_upload(new_path, f"sft_step_{step}.pt")
        self._hf_prune()

        while len(self._local) > self.keep_local_n:
            oldest = self._local.pop(0)
            if os.path.isfile(oldest): os.remove(oldest)
            print(f"[CKPT] Local evict: {Path(oldest).name}", flush=True)

    def _hf_upload(self, local_path, repo_filename):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=self.hf_repo, token=self.hf_token,
                            repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=local_path,
                            path_in_repo=repo_filename,
                            repo_id=self.hf_repo, token=self.hf_token)
            print(f"[HF] Uploaded {repo_filename}", flush=True)
        except Exception as e:
            print(f"[HF] Upload failed: {e}", flush=True)

    def _hf_prune(self):
        try:
            from huggingface_hub import HfApi
            api   = HfApi()
            files = [f.rfilename for f in
                     api.list_repo_files(repo_id=self.hf_repo, token=self.hf_token)]
            ckpts = sorted(
                [f for f in files if f.startswith("sft_step_") and f.endswith(".pt")],
                key=lambda x: int(x.replace("sft_step_","").replace(".pt",""))
            )
            for fname in ckpts[:-self.keep_hf_n]:
                api.delete_file(path_in_repo=fname,
                                repo_id=self.hf_repo, token=self.hf_token)
                print(f"[HF] Deleted: {fname}", flush=True)
        except Exception as e:
            print(f"[HF] Prune failed: {e}", flush=True)


# ── Args ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_ckpt", type=str, default=None,
                   help="Path to pretrained .pt file (auto-downloads from HF if absent)")
    p.add_argument("--pretrain_repo", type=str, default="shawneil/NanoMind")
    p.add_argument("--d_model",       type=int,   default=512)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--n_kv_heads",    type=int,   default=2)
    p.add_argument("--n_layers",      type=int,   default=8)
    p.add_argument("--max_seq_len",   type=int,   default=512)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--accum_steps",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--max_epochs",    type=int,   default=3)
    p.add_argument("--warmup_steps",  type=int,   default=100)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--compile",       action="store_true")
    p.add_argument("--keep_local_n",  type=int,   default=3)
    p.add_argument("--keep_hf_n",     type=int,   default=2)
    p.add_argument("--save_every",    type=int,   default=500)
    p.add_argument("--log_every",     type=int,   default=10)
    p.add_argument("--out_dir",       type=str,   default="./sft_checkpoints")
    p.add_argument("--hf_token",      type=str,   default=None)
    p.add_argument("--hf_repo_id",    type=str,   default="shawneil/NanoMind-SFT")
    p.add_argument("--wandb_project", type=str,   default="nano-nanomind-sft")
    p.add_argument("--wandb_api_key", type=str,   default=None)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args                    = parse_args()
    rank, local_rank, world = init_ddp()
    device                  = torch.device(f"cuda:{local_rank}")
    is_main                 = (rank == 0)

    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
    scaler  = torch.cuda.amp.GradScaler(enabled=True)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # ── WandB (rank 0 only) ──────────────────────────────────────────
    wandb = None
    if is_main:
        import wandb as _wb
        _wb.login(key=args.wandb_api_key or os.environ.get("WANDB_API_KEY", ""),
                  relogin=True)
        _wb.init(project=args.wandb_project, config=vars(args), resume="allow")
        wandb = _wb

    # ── Model ────────────────────────────────────────────────────────
    from model import ModelConfig, NanoMind
    cfg = ModelConfig(
        vocab_size=enc.n_vocab, d_model=args.d_model,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers, max_seq_len=args.max_seq_len,
    )
    model = NanoMind(cfg).to(device)

    # ── Load pretrained weights ──────────────────────────────────────
    ckpt_path = args.pretrain_ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        # try to find one locally first
        existing = sorted(
            Path("/kaggle/working/checkpoints").glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        ) if Path("/kaggle/working/checkpoints").exists() else []
        if existing:
            ckpt_path = str(existing[-1])
            if is_main: print(f"[INIT] Using local pretrained: {ckpt_path}", flush=True)
        else:
            # download latest from HF
            if is_main:
                print(f"[INIT] Downloading latest checkpoint from {args.pretrain_repo}...", flush=True)
                from huggingface_hub import hf_hub_download, list_repo_files
                hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
                files = [f for f in list_repo_files(args.pretrain_repo, token=hf_token)
                         if f.startswith("checkpoint_step_") and f.endswith(".pt")]
                latest = sorted(files, key=lambda x: int(x.split("_")[-1].replace(".pt","")))[- 1]
                os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
                ckpt_path = hf_hub_download(
                    repo_id=args.pretrain_repo, filename=latest,
                    token=hf_token, local_dir="/kaggle/working/checkpoints"
                )
                print(f"[INIT] Downloaded: {latest}", flush=True)
            # broadcast path string to all ranks
            path_list = [ckpt_path] if is_main else [None]
            dist.broadcast_object_list(path_list, src=0)
            ckpt_path = path_list[0]

    # ── Dataset (build early so we know steps_per_epoch for LR schedule) ──
    from sft_data import build_alpaca_dataset, AlpacaDataset, collate_fn
    samples = build_alpaca_dataset(enc, args.max_seq_len, rank=rank, world_size=world)
    dataset = AlpacaDataset(samples)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    if is_main:
        print(f"[DATA] {len(dataset)} samples/rank, {len(loader)} batches/epoch", flush=True)

    # ── Optimizer ────────────────────────────────────────────────────
    decay   = [p for n,p in model.named_parameters() if p.dim()>=2 and p.requires_grad]
    nodecay = [p for n,p in model.named_parameters() if p.dim()< 2 and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": decay,   "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, fused=True,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Resume: SFT ckpt takes priority over pretrained base ─────────
    start_step  = 0
    start_epoch = 0
    sft_existing = sorted(
        Path(args.out_dir).glob("sft_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if sft_existing:
        # CONTINUE from previous SFT run — do NOT reload pretrained base
        resume_path = str(sft_existing[-1])
        if is_main:
            print(f"[RESUME] Found SFT checkpoint: {Path(resume_path).name}", flush=True)
        sft_ckpt = torch.load(resume_path,
                              map_location={"cuda:0": f"cuda:{local_rank}"})
        model.load_state_dict(sft_ckpt["model"], strict=True)
        optimizer.load_state_dict(sft_ckpt["optimizer"])
        start_step  = sft_ckpt.get("step", 0)
        start_epoch = sft_ckpt.get("epoch", 0)
        if is_main:
            print(f"[RESUME] Continuing from epoch {start_epoch+1}, "
                  f"global step {start_step}", flush=True)
    else:
        # Fresh SFT start — load pretrained base weights
        if is_main:
            print(f"[INIT] No SFT checkpoint found — loading pretrained base", flush=True)
        ckpt = torch.load(ckpt_path, map_location={"cuda:0": f"cuda:{local_rank}"})
        model.load_state_dict(ckpt["model"], strict=True)
        if is_main:
            print(f"[INIT] Loaded pretrained weights (step={ckpt.get('step','?')})", flush=True)

    if is_main:
        print(f"[INIT] Model params: {model.num_params()/1e6:.2f}M", flush=True)

    # ── torch.compile ────────────────────────────────────────────────
    if args.compile:
        if is_main: print("torch.compile(mode='default')...", flush=True)
        model = torch.compile(model, mode="default")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    ckpt_mgr = CheckpointManager(
        out_dir=args.out_dir, keep_local_n=args.keep_local_n,
        keep_hf_n=args.keep_hf_n,
        hf_repo=args.hf_repo_id, hf_token=hf_token,
    ) if is_main else None

    # ── total steps for LR schedule ──────────────────────────────────
    steps_per_epoch = len(loader) // args.accum_steps
    total_steps     = steps_per_epoch * args.max_epochs   # full 5-epoch budget
    # warmup only applies if we're still in warmup range
    effective_warmup = max(0, args.warmup_steps - start_step)
    if is_main:
        print(f"[TRAIN] {steps_per_epoch} steps/epoch × {args.max_epochs} epochs "
              f"= {total_steps} total optimizer steps", flush=True)
        print(f"[TRAIN] Resuming at step {start_step}/{total_steps} "
              f"(epoch {start_epoch+1}→{args.max_epochs})", flush=True)
        if effective_warmup > 0:
            print(f"[TRAIN] Remaining warmup steps: {effective_warmup}", flush=True)

    global_step = start_step
    model.train()
    t0 = time.perf_counter()

    for epoch in range(start_epoch, args.max_epochs):
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        for x, y, mask in loader:
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            ctx = model.no_sync() if (micro % args.accum_steps) < args.accum_steps - 1 \
                  else nullctx()
            with ctx:
                with amp_ctx:
                    _, loss = model(x, y, loss_mask=mask)
                    loss    = loss / args.accum_steps
                scaler.scale(loss).backward()

            micro += 1
            if micro % args.accum_steps != 0:
                continue

            # — optimizer step —
            lr = cosine_lr(global_step, args.warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_val     = loss.item() * args.accum_steps

            if is_main and global_step % args.log_every == 0:
                dt   = (time.perf_counter() - t0) / args.log_every * 1000
                t0   = time.perf_counter()
                toks = args.batch_size * args.accum_steps * args.max_seq_len * world
                print(f"epoch {epoch+1} | step {global_step:>6} | "
                      f"loss {loss_val:.4f} | lr {lr:.2e} | "
                      f"{dt:.1f} ms/step | {toks/dt*1000/1e6:.2f}M tok/s", flush=True)
                if wandb:
                    wandb.log({"sft/loss": loss_val, "sft/lr": lr,
                               "sft/epoch": epoch + 1,
                               "sft/tok_per_sec": toks / dt * 1000},
                              step=global_step)

            if is_main and global_step % args.save_every == 0:
                ckpt_mgr.save(model, optimizer, global_step, epoch)

        if is_main:
            print(f"[EPOCH {epoch+1} DONE] step={global_step}", flush=True)
            ckpt_mgr.save(model, optimizer, global_step, epoch + 1)

    cleanup()
    if is_main:
        print("SFT complete.", flush=True)
        if wandb: wandb.finish()

if __name__ == "__main__":
    main()
