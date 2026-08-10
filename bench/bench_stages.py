"""Where does the step time actually go? Forward-only timing of each stage,
plus the memory traffic of the masking approach."""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChessNet, POLICY_SIZE

B = int(os.environ.get("BENCH_B", "256"))
ITERS = 20


def timed(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    dev = torch.device("cuda")
    m = ChessNet().to(dev)
    g = torch.Generator().manual_seed(0)
    pieces = torch.randint(0, 13, (B, 64), generator=g).to(dev)
    aux = torch.stack([torch.randint(0, 32, (B,), generator=g),
                       torch.randint(0, 16, (B,), generator=g)], dim=1).to(dev)
    targets = torch.randint(0, POLICY_SIZE, (B,), generator=g).to(dev)
    masks = torch.zeros(B, POLICY_SIZE, dtype=torch.bool, device=dev)
    masks.scatter_(1, torch.randint(0, POLICY_SIZE, (B, 19), generator=g).to(dev), True)
    masks.scatter_(1, targets[:, None], True)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        trunk = timed(lambda: m(pieces, aux))
        logits, _ = m(pieces, aux)
    logits = logits.float()

    def mask_where():
        x = logits + torch.where(masks, torch.zeros_like(logits),
                                 torch.full_like(logits, -1e9))
        return F.cross_entropy(x, targets)

    def mask_fill():
        return F.cross_entropy(logits.masked_fill(~masks, -1e9), targets)

    print(f"batch {B}")
    print(f"  full forward (bf16, no_grad)   {trunk:7.2f} ms")
    print(f"  mask via torch.where + CE      {timed(mask_where):7.2f} ms")
    print(f"  mask via masked_fill + CE      {timed(mask_fill):7.2f} ms")

    # parameter / FLOP accounting
    d, L = m.d, len(m.blocks)
    seq = 67
    attn_proj = 4 * d * d
    mlp = 8 * d * d
    per_layer = (attn_proj + mlp) * 2 * seq          # mul-add, per sample
    trunk_flops = per_layer * L
    head_flops = 2 * 64 * d * 73
    print(f"\n  trunk FLOPs/sample  {trunk_flops/1e6:8.1f} M")
    print(f"  head  FLOPs/sample  {head_flops/1e6:8.1f} M")
    print(f"  logits tensor       {B*POLICY_SIZE*4/1e6:8.1f} MB fp32 "
          f"(x{4} for where+zeros_like+full_like+out)")
    print(f"  params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")


if __name__ == "__main__":
    main()
