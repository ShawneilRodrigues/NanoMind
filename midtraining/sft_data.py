"""
Alpaca SFT Dataset
- Formats each example as:  ### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}<|endoftext|>
- loss_mask = 1 ONLY on response tokens  (instruction/input tokens are ignored in loss)
- Pre-tokenises the full dataset once into a list of (input_ids, targets, loss_mask)
- Shards across DDP ranks by index
"""

import torch
from torch.utils.data import Dataset

PROMPT_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

def build_alpaca_dataset(enc, max_seq_len, rank=0, world_size=1):
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split="train")

    eot = enc.eot_token
    samples = []

    for ex in raw:
        instruction = ex["instruction"].strip()
        inp         = ex.get("input", "").strip()
        output      = ex["output"].strip()

        if inp:
            prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
        else:
            prompt = PROMPT_NO_INPUT.format(instruction=instruction)

        prompt_ids   = enc.encode_ordinary(prompt)
        response_ids = enc.encode_ordinary(output) + [eot]

        full_ids = prompt_ids + response_ids
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        # loss mask: 0 for prompt tokens, 1 for response tokens
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        mask = mask[:max_seq_len]

        # pad to max_seq_len
        pad_len  = max_seq_len - len(full_ids)
        full_ids = full_ids + [-1] * pad_len   # -1 → ignore_index in CE
        mask     = mask     + [0]  * pad_len

        input_ids = full_ids[:-1]
        targets   = full_ids[1:]
        lmask     = mask[1:]          # align with targets

        # skip sample if response portion is entirely truncated away
        if sum(lmask) == 0:
            continue

        samples.append((
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets,   dtype=torch.long),
            torch.tensor(lmask,     dtype=torch.bool),
        ))

    # shard
    samples = samples[rank::world_size]
    print(f"[rank {rank}] SFT samples: {len(samples)}", flush=True)
    return samples


class AlpacaDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    xs   = torch.stack([b[0] for b in batch])
    ys   = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    return xs, ys, mask
