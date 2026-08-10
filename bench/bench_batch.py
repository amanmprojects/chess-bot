"""Throughput vs batch size for the current model. Small models on modern GPUs
are usually launch-bound at small batch: rec/s should climb with B until the
GPU saturates. Reports rec/s so batches are directly comparable."""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChessNet, POLICY_SIZE

ITERS = 12
SIZES = [int(x) for x in os.environ.get(
    "SIZES", "256,512,1024,2048,4096").split(",")]


def bench(B, dev):
    m = ChessNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.1, fused=True)
    g = torch.Generator().manual_seed(0)
    pieces = torch.randint(0, 13, (B, 64), generator=g).to(dev)
    aux = torch.stack([torch.randint(0, 32, (B,), generator=g),
                       torch.randint(0, 16, (B,), generator=g)], dim=1).to(dev)
    targets = torch.randint(0, POLICY_SIZE, (B,), generator=g).to(dev)
    vals = torch.randint(-1, 2, (B,), generator=g).float().to(dev)
    masks = torch.zeros(B, POLICY_SIZE, dtype=torch.bool, device=dev)
    masks.scatter_(1, torch.randint(0, POLICY_SIZE, (B, 19), generator=g).to(dev), True)
    masks.scatter_(1, targets[:, None], True)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, pred = m(pieces, aux)
            logits = logits.float()
            ce = F.cross_entropy(logits.masked_fill(~masks, -1e9), targets)
            loss = ce + 0.5 * F.mse_loss(pred, vals)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / ITERS * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return ms, B / (ms / 1000), peak


def main():
    dev = torch.device("cuda")
    print(f"{'batch':>6} {'ms/step':>9} {'rec/s':>10} {'peak GB':>9}")
    for B in SIZES:
        try:
            ms, rate, peak = bench(B, dev)
            print(f"{B:>6} {ms:>9.1f} {rate:>10.0f} {peak:>9.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>6} {'OOM':>9}", flush=True)
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
