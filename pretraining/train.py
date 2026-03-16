import os, sys, math, time, json, argparse, shutil
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True

def build_enc():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc.n_vocab

def enc_encode(enc, text):
    return enc.encode_ordinary(text)

def enc_decode(enc, ids):
    return enc.decode(ids)

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


# ══════════════════════════════════════════════════════════════════════
# Streaming dataset
# ══════════════════════════════════════════════════════════════════════
class StreamingTokenDataset:
    INDEX_VERSION = 1

    def __init__(self, enc, dataset_name, text_field,
                 seq_len, rank, world_size, index_path):
        self.enc          = enc
        self.dataset_name = dataset_name
        self.text_field   = text_field
        self.seq_len      = seq_len
        self.rank         = rank
        self.world_size   = world_size
        self.index_path   = index_path
        self.global_doc   = 0
        self.buf          = []
        self.buf_offset   = 0
        self._load_index()

    def _load_index(self):
        if os.path.isfile(self.index_path):
            try:
                s = json.loads(Path(self.index_path).read_text())
                if s.get("version") == self.INDEX_VERSION:
                    self.global_doc = s.get("global_doc", 0)
                    print(f"[rank {self.rank}] Stream index: "
                          f"resuming from doc {self.global_doc}", flush=True)
            except Exception as e:
                print(f"[rank {self.rank}] Stream index load failed: {e}", flush=True)

    def save_index(self):
        tmp = self.index_path + ".tmp"
        Path(tmp).write_text(json.dumps({
            "version":    self.INDEX_VERSION,
            "global_doc": self.global_doc,
        }))
        os.replace(tmp, self.index_path)

    def _open_stream(self):
        from datasets import load_dataset
        hf_tok = os.environ.get("HF_TOKEN", "")
        ds = load_dataset(
            self.dataset_name, split="train",
            streaming=True, trust_remote_code=True,
            **({"token": hf_tok} if hf_tok else {}),
        )
        return iter(ds)

    def _fill(self, it):
        eot = self.enc.eot_token
        while len(self.buf) - self.buf_offset < self.seq_len + 1:
            try:
                doc = next(it)
            except StopIteration:
                return False
            self.buf.extend(enc_encode(self.enc, doc[self.text_field]))
            self.buf.append(eot)
            self.global_doc += 1
            if len(self.buf) > 2_000_000:
                self.buf        = self.buf[self.buf_offset:]
                self.buf_offset = 0
        return True

    def __iter__(self):
        it         = self._open_stream()
        doc_cursor = 0
        target     = self.global_doc
        if target > 0:
            print(f"[rank {self.rank}] Fast-skipping to doc {target}...", flush=True)
        while doc_cursor < target:
            try:
                next(it); doc_cursor += 1
            except StopIteration:
                self.global_doc = 0; doc_cursor = 0
                self.buf = []; self.buf_offset = 0
                it = self._open_stream()
        print(f"[rank {self.rank}] Dataset ready at doc {self.global_doc}", flush=True)
        while True:
            if not self._fill(it):
                self.global_doc = 0; self.buf = []; self.buf_offset = 0
                it = self._open_stream()
                continue
            end   = self.buf_offset + self.seq_len + 1
            chunk = torch.tensor(self.buf[self.buf_offset:end], dtype=torch.long)
            self.buf_offset += self.seq_len
            yield chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════════════════════════════
# CheckpointManager  —  LOCAL: keep_local_n  |  HF: keep_hf_n
# ══════════════════════════════════════════════════════════════════════
class CheckpointManager:
    def __init__(self, out_dir, keep_local_n, keep_hf_n, hf_repo, hf_token):
        self.out_dir      = out_dir
        self.keep_local_n = keep_local_n
        self.keep_hf_n    = keep_hf_n
        self.hf_repo      = hf_repo
        self.hf_token     = hf_token
        self._local = sorted(
            [str(p) for p in Path(out_dir).glob("checkpoint_step_*.pt")],
            key=lambda x: int(Path(x).stem.split("_")[-1])
        )

    def save(self, model, optimizer, step, index_files):
        raw = model.module
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod

        new_path = os.path.join(self.out_dir, f"checkpoint_step_{step}.pt")

        # pre-save disk guard
        free_mb = shutil.disk_usage(self.out_dir).free / 1024 / 1024
        if free_mb < 400 and self._local:
            oldest = self._local.pop(0)
            if os.path.isfile(oldest):
                os.remove(oldest)
                print(f"[CKPT] Pre-save evict (low disk): {Path(oldest).name}", flush=True)

        torch.save({
            "step":      step,
            "model":     raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    vars(raw.cfg),
        }, new_path)
        size_mb = os.path.getsize(new_path) / 1024 / 1024
        print(f"[CKPT] Saved {Path(new_path).name} ({size_mb:.1f} MB)", flush=True)
        self._local.append(new_path)

        # upload new checkpoint to HF
        self._hf_upload(new_path, f"checkpoint_step_{step}.pt")

        # upload stream indexes (always overwrite in-place — never accumulate)
        for idx_path in index_files:
            if os.path.isfile(idx_path):
                self._hf_upload(idx_path, os.path.basename(idx_path))

        # prune HF: keep only latest keep_hf_n checkpoint .pt files
        self._hf_prune_checkpoints()

        # prune local: keep only latest keep_local_n
        while len(self._local) > self.keep_local_n:
            oldest = self._local.pop(0)
            if os.path.isfile(oldest):
                os.remove(oldest)
                print(f"[CKPT] Local evict: {Path(oldest).name}", flush=True)

        print(f"[CKPT] Local ({len(self._local)}/{self.keep_local_n}): "
              + ", ".join(Path(p).name for p in self._local), flush=True)

    def _hf_upload(self, local_path, repo_filename):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=self.hf_repo, token=self.hf_token,
                            repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_filename,
                repo_id=self.hf_repo,
                token=self.hf_token,
            )
            print(f"[HF] Uploaded {repo_filename} → {self.hf_repo}", flush=True)
        except Exception as e:
            print(f"[HF] Upload failed for {repo_filename}: {e}", flush=True)

    def _hf_prune_checkpoints(self):
        """Delete checkpoint .pt files on HF beyond keep_hf_n. JSONs are never deleted."""
        try:
            from huggingface_hub import HfApi
            api   = HfApi()
            files = [
                f.rfilename
                for f in api.list_repo_files(
                    repo_id=self.hf_repo, token=self.hf_token
                )
            ]
            ckpt_files = sorted(
                [f for f in files
                 if f.startswith("checkpoint_step_") and f.endswith(".pt")],
                key=lambda x: int(x.replace("checkpoint_step_","").replace(".pt",""))
            )
            to_delete = ckpt_files[:-self.keep_hf_n] if len(ckpt_files) > self.keep_hf_n else []
            for fname in to_delete:
                api.delete_file(path_in_repo=fname,
                                repo_id=self.hf_repo, token=self.hf_token)
                print(f"[HF] Deleted old: {fname}", flush=True)
            kept = ckpt_files[-self.keep_hf_n:]
            print(f"[HF] Keeping ({len(kept)}/{self.keep_hf_n}): "
                  + ", ".join(kept), flush=True)
        except Exception as e:
            print(f"[HF] Prune failed (non-fatal): {e}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════════
def log_sample(model, enc, device, prompt, max_new, step, wandb):
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    with torch.no_grad():
        idx = torch.tensor([enc_encode(enc, prompt)],
                           dtype=torch.long, device=device)
        out = raw.generate(idx, max_new_tokens=max_new)
    text = enc_decode(enc, out[0].tolist())
    wandb.log({"sample/text": wandb.Html(f"<pre>{text}</pre>"),
               "sample/step": step}, step=step)
    print(f"[sample @ {step}]\n{text[:300]}\n", flush=True)
    raw.train()

@contextmanager
def nullctx():
    yield


# ══════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_kv_heads",     type=int,   default=2)
    p.add_argument("--n_layers",       type=int,   default=8)
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--use_moe",        action="store_true")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--accum_steps",    type=int,   default=8)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=0.1)
    p.add_argument("--max_steps",      type=int,   default=500_000)
    p.add_argument("--warmup_steps",   type=int,   default=500)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--use_8bit_adam",  action="store_true")
    p.add_argument("--compile",        action="store_true")
    p.add_argument("--grad_ckpt",      action="store_true")
    p.add_argument("--keep_local_n",   type=int,   default=5)
    p.add_argument("--keep_hf_n",      type=int,   default=2)
    p.add_argument("--dataset",        type=str,   default="Skylion007/openwebtext")
    p.add_argument("--text_field",     type=str,   default="text")
    p.add_argument("--out_dir",        type=str,   default="./checkpoints")
    p.add_argument("--save_every",     type=int,   default=1500)
    p.add_argument("--resume_from",    type=str,   default=None)
    p.add_argument("--hf_token",       type=str,   default=None)
    p.add_argument("--hf_repo_id",     type=str,   default="shawneil/NanoMind")
    p.add_argument("--wandb_project",  type=str,   default="nano-nanomind")
    p.add_argument("--wandb_api_key",  type=str,   default=None)
    p.add_argument("--log_every",      type=int,   default=10)
    p.add_argument("--sample_every",   type=int,   default=1000)
    p.add_argument("--sample_prompt",  type=str,   default="Once upon a time")
    p.add_argument("--sample_max_new", type=int,   default=200)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    args                    = parse_args()
    rank, local_rank, world = init_ddp()
    device                  = torch.device(f"cuda:{local_rank}")
    is_main                 = (rank == 0)

    amp_dtype = torch.float16
    amp_ctx   = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=True)
    scaler    = torch.cuda.amp.GradScaler(enabled=True)

    enc, vocab_size = build_enc()

    wandb = None
    if is_main:
        import wandb as _wb
        _wb.login(key=args.wandb_api_key or os.environ.get("WANDB_API_KEY",""),
                  relogin=True)
        _wb.init(project=args.wandb_project, config=vars(args), resume="allow")
        wandb = _wb

    from model import ModelConfig, NanoMind
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers, max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
    )
    model = NanoMind(cfg).to(device)

    if args.grad_ckpt:
        from torch.utils.checkpoint import checkpoint
        for blk in model.blocks:
            _orig = blk.forward
            def _ckpt(fn):
                def _w(*a, **kw): return checkpoint(fn, *a, use_reentrant=False, **kw)
                return _w
            blk.forward = _ckpt(_orig)

    if is_main:
        print(f"Model: {model.num_params()/1e6:.2f}M params", flush=True)

    decay   = [p for n,p in model.named_parameters() if p.dim()>=2 and p.requires_grad]
    nodecay = [p for n,p in model.named_parameters() if p.dim()< 2 and p.requires_grad]
    groups  = [{"params": decay,   "weight_decay": args.weight_decay},
               {"params": nodecay, "weight_decay": 0.0}]

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, lr=args.lr)
        except Exception:
            optimizer = torch.optim.AdamW(groups, lr=args.lr, fused=True)
    else:
        optimizer = torch.optim.AdamW(groups, lr=args.lr, fused=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    start_step   = 0
    ckpt_to_load = args.resume_from
    if not ckpt_to_load:
        existing = sorted(Path(args.out_dir).glob("checkpoint_step_*.pt"),
                          key=lambda p: int(p.stem.split("_")[-1]))
        if existing:
            ckpt_to_load = str(existing[-1])
            if is_main: print(f"Auto-resuming: {ckpt_to_load}", flush=True)

    if ckpt_to_load and os.path.isfile(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load,
                          map_location={"cuda:0": f"cuda:{local_rank}"})
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if is_main: print(f"Resumed at step {start_step}", flush=True)

    if args.compile:
        if is_main: print("torch.compile(mode='default')...", flush=True)
        model = torch.compile(model, mode="default")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    hf_repo  = args.hf_repo_id or os.environ.get("HF_REPO_ID", "")
    hf_token = args.hf_token   or os.environ.get("HF_TOKEN",   "")
    ckpt_mgr = CheckpointManager(
        out_dir=args.out_dir, keep_local_n=args.keep_local_n,
        keep_hf_n=args.keep_hf_n, hf_repo=hf_repo, hf_token=hf_token,
    ) if is_main else None

    index_path = os.path.join(args.out_dir, f"stream_index_rank{rank}.json")
    ds = StreamingTokenDataset(
        enc=enc, dataset_name=args.dataset, text_field=args.text_field,
        seq_len=args.max_seq_len, rank=rank, world_size=world,
        index_path=index_path,
    )
    ds_iter = iter(ds)

    def next_batch():
        xs, ys = [], []
        for _ in range(args.batch_size):
            x, y = next(ds_iter)
            xs.append(x); ys.append(y)
        return (torch.stack(xs).to(device, non_blocking=True),
                torch.stack(ys).to(device, non_blocking=True))

    model.train()
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    t0   = time.perf_counter()

    while step < args.max_steps:
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for micro in range(args.accum_steps):
            x, y = next_batch()
            ctx  = model.no_sync() if micro < args.accum_steps-1 else nullctx()
            with ctx:
                with amp_ctx:
                    _, loss = model(x, y)
                    loss    = loss / args.accum_steps
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step    += 1
        loss_val = loss.item() * args.accum_steps

        if is_main and step % args.log_every == 0:
            dt   = (time.perf_counter() - t0) / args.log_every * 1000
            t0   = time.perf_counter()
            toks = args.batch_size * args.accum_steps * args.max_seq_len * world
            print(f"step {step:>7} | loss {loss_val:.4f} | lr {lr:.2e} "
                  f"| {dt:.1f} ms/step | {toks/dt*1000/1e6:.2f}M tok/s", flush=True)
            if wandb:
                wandb.log({"train/loss": loss_val, "train/lr": lr,
                           "train/tok_per_sec": toks/dt*1000}, step=step)

        if is_main and wandb and step % args.sample_every == 0:
            log_sample(model, enc, device,
                       args.sample_prompt, args.sample_max_new, step, wandb)

        if step % args.save_every == 0:
            ds.buf = []; ds.buf_offset = 0
            ds.save_index()
            if is_main:
                index_files = [
                    os.path.join(args.out_dir, "stream_index_rank0.json"),
                    os.path.join(args.out_dir, "stream_index_rank1.json"),
                ]
                ckpt_mgr.save(model, optimizer, step, index_files)

    cleanup()
    if is_main:
        print("Training complete.", flush=True)
        if wandb: wandb.finish()

if __name__ == "__main__":
    main()
