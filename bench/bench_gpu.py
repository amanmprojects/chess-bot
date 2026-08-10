"""Micro-benchmark of the train step on GPU, and of candidate optimizations.

NOTE: run this while nothing else uses the GPU for clean numbers. If a training
job is live, treat the absolute ms as inflated but the ratios as meaningful.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChessNet, SEQ_LEN, NUM_SQUARES, POLICY_SIZE

B = int(os.environ.get("BENCH_B", "1024"))
WARMUP = 5
ITERS = 20


def timed(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def fake_batch(device):
    g = torch.Generator(device="cpu").manual_seed(0)
    pieces = torch.randint(0, 13, (B, 64), generator=g).to(device)
    aux = torch.stack([torch.randint(0, 32, (B,), generator=g),
                       torch.randint(0, 16, (B,), generator=g)], dim=1).to(device)
    targets = torch.randint(0, POLICY_SIZE, (B,), generator=g).to(device)
    vals = torch.randint(-1, 2, (B,), generator=g).float().to(device)
    masks = torch.zeros(B, POLICY_SIZE, dtype=torch.bool, device=device)
    idx = torch.randint(0, POLICY_SIZE, (B, 19), generator=g).to(device)
    masks.scatter_(1, idx, True)
    masks.scatter_(1, targets[:, None], True)
    return pieces, aux, targets, vals, masks


def make_step(model, opt, scaler, batch, mode):
    pieces, aux, targets, vals, masks = batch
    neg = -1e9

    def step():
        opt.zero_grad(set_to_none=True)
        if mode == "bf16":
            ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            ctx = torch.autocast("cuda", dtype=torch.float16)
        with ctx:
            logits, pred = model(pieces, aux)
            logits = logits.float()
            if mode == "baseline" or mode == "bf16":
                masked = logits + torch.where(masks, torch.zeros_like(logits),
                                              torch.full_like(logits, neg))
            else:
                masked = logits.masked_fill(~masks, neg)
            ce = F.cross_entropy(masked, targets)
            loss = ce + 0.5 * F.mse_loss(pred, vals)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return step


class SDPABlock(nn.Module):
    """Same math as model.Block, but fused QKV + F.scaled_dot_product_attention."""

    def __init__(self, d, heads, mlp_scale):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_scale), nn.GELU(),
                                 nn.Linear(d * mlp_scale, d))

    def forward(self, x):
        Bs, T, d = x.shape
        n = self.ln1(x)
        q, k, v = self.qkv(n).split(d, dim=2)
        q = q.view(Bs, T, self.h, self.dh).transpose(1, 2)
        k = k.view(Bs, T, self.h, self.dh).transpose(1, 2)
        v = v.view(Bs, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).contiguous().view(Bs, T, d)
        x = x + self.proj(o)
        x = x + self.mlp(self.ln2(x))
        return x


def swap_sdpa(model):
    for i, blk in enumerate(model.blocks):
        d = blk.ln1.normalized_shape[0]
        heads = blk.attn.num_heads
        mlp_scale = blk.mlp[0].out_features // d
        new = SDPABlock(d, heads, mlp_scale).to(next(model.parameters()).device)
        model.blocks[i] = new
    return model


def build(device, sdpa=False, channels_last=False):
    m = ChessNet().to(device)
    if sdpa:
        m = swap_sdpa(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    batch = fake_batch(device)
    results = {}

    def run(name, model, mode, scaler_on=True, compile_=False, fused=False):
        if args.only and args.only not in name:
            return
        if compile_:
            model = torch.compile(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1,
                                fused=fused)
        scaler = torch.amp.GradScaler("cuda") if scaler_on else None
        try:
            ms = timed(make_step(model, opt, scaler, batch, mode))
            results[name] = ms
            print(f"{name:<34} {ms:8.1f} ms  {B/(ms/1000):9.0f} rec/s", flush=True)
        except Exception as e:
            print(f"{name:<34} FAILED: {type(e).__name__}: {e}", flush=True)
        del opt, model
        torch.cuda.empty_cache()

    print(f"batch {B}, seq {SEQ_LEN}\n")
    run("1 baseline (fp16+scaler, MHA)", build(device), "baseline")
    run("2 + masked_fill", build(device), "maskfill")
    run("3 + sdpa attention", build(device, sdpa=True), "maskfill")
    run("4 + bf16 (no scaler)", build(device, sdpa=True), "bf16", scaler_on=False)
    run("5 + torch.compile", build(device, sdpa=True), "bf16",
        scaler_on=False, compile_=True)
    run("6 + fused adamw", build(device, sdpa=True), "bf16",
        scaler_on=False, compile_=True, fused=True)

    if len(results) > 1:
        base = results.get("1 baseline (fp16+scaler, MHA)")
        if base:
            print()
            for k, v in results.items():
                print(f"  {k:<34} {base/v:5.2f}x")


if __name__ == "__main__":
    main()
