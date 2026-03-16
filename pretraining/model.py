import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size:    int   = 50257
    d_model:       int   = 512
    n_heads:       int   = 8
    n_kv_heads:    int   = 2
    n_layers:      int   = 8
    max_seq_len:   int   = 512
    ff_mult:       int   = 4
    dropout:       float = 0.0
    use_moe:       bool  = False
    num_experts:   int   = 4
    top_k_experts: int   = 2

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(x.dtype).clone() * self.weight

def precompute_freqs_cis(head_dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(xq, xk, freqs_cis):
    def rot(x, f):
        xc  = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(xc * f[:x.shape[1]].unsqueeze(0).unsqueeze(2)).flatten(3)
        return out.to(x.dtype)
    return rot(xq, freqs_cis), rot(xk, freqs_cis)

class GQAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh        = cfg.n_heads
        self.nkv       = cfg.n_kv_heads
        self.hd        = cfg.d_model // cfg.n_heads
        self.q         = nn.Linear(cfg.d_model, cfg.n_heads    * self.hd, bias=False)
        self.k         = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v         = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o         = nn.Linear(cfg.n_heads * self.hd, cfg.d_model,    bias=False)
        self.attn_drop = cfg.dropout
    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh,  self.hd)
        k = self.k(x).view(B, T, self.nkv, self.hd)
        v = self.v(x).view(B, T, self.nkv, self.hd)
        q, k = apply_rope(q, k, freqs_cis)
        reps  = self.nh // self.nkv
        k = k.repeat_interleave(reps, dim=2)
        v = v.repeat_interleave(reps, dim=2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )
        return self.o(out.transpose(1,2).contiguous().view(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = int(cfg.d_model * cfg.ff_mult * 2 / 3)
        h = (h + 63) // 64 * 64
        self.w1 = nn.Linear(cfg.d_model, h, bias=False)
        self.w2 = nn.Linear(h, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k   = cfg.top_k_experts
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(cfg.num_experts)])
        self.gate    = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        flat    = x.view(-1, D)
        w, idx  = torch.topk(F.softmax(self.gate(flat), dim=-1), self.top_k, dim=-1)
        out     = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            sel = (idx == i).any(dim=-1)
            if not sel.any(): continue
            rows = sel.nonzero(as_tuple=True)[0]
            kpos = (idx[rows] == i).nonzero(as_tuple=True)[1]
            out[rows] += w[rows, kpos].unsqueeze(-1) * expert(flat[rows])
        return out.view(B, T, D)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.an   = RMSNorm(cfg.d_model)
        self.fn   = RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg)
        self.ff   = MoEFF(cfg) if cfg.use_moe else SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x, fc):
        x = x + self.drop(self.attn(self.an(x), fc))
        x = x + self.drop(self.ff(self.fn(x)))
        return x

class NanoMind(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.d_model // cfg.n_heads, cfg.max_seq_len * 2)
        )
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x    = self.drop(self.embed(idx))
        fc   = self.freqs_cis[:T]
        for blk in self.blocks:
            x = blk(x, fc)
        x      = self.norm(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            ic = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ic)
            logits = logits[:, -1, :] / temperature
            v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat(
                [idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1
            )
        return idx
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
