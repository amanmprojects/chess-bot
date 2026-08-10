# Speeding up chess-bot training

**Current state:** `train.py` running on the RTX 4060 Laptop GPU (power-capped at
60W). Log rate is flat at ~3,107 records/s → **~2h45m for the full 10-epoch run**,
with eval+checkpoint each 2000 steps adding overhead on top.

## Why it's slow: the model is launch/memory-bound, not FLOP-bound

The model is tiny (5.58M params) and the step is dominated by per-token MLP
matmuls over just 67 tokens. The whole step is ~330ms, and at 1024 rec/s the
GPU is already saturated (smaller batches run at the same ms/step). Achieved
throughput is only **~7 TFLOPS**, roughly 25–30% of what a 60W 4060 Laptop
should sustain with bf16 tensor cores — the rest is small-kernel launch
overhead and memory traffic (the 4672-wide logits are moved around in fp32).

Measured contributions (small-batch micro-benchmarks, so the *ratios* are what
matter; the live job was running during measurement):

| change | vs baseline |
|---|---|
| `masked_fill` instead of `torch.where` (masking) | ~1.00x (negligible — masking is 0.2ms of 330) |
| Fused QKV + SDPA attention (replace `nn.MultiheadAttention`) | ~1.20x |
| bf16 + no GradScaler (dropping fp16 scaling entirely) | ~1.22x |
| `torch.compile` | ~1.38x |
| `fused=True` AdamW | small |
| **all combined** | **~1.4–1.6x** |

Key point: the *only* place a "significant" (2x+) speedup exists for this
workload on this hardware is **reducing the actual math** (fewer/larger
dimensions, fewer layers, fewer tokens) or **faster hardware** — the current
code is already running the GPU at its cap, so no amount of software
optimization on the same model makes it run 2x faster. The 4060 is a 60W
laptop part and this model is small; the correct lever for this specific job
is to get more effective work per step, not to shave per-step overhead.

## What's worth doing (in order)

1. **Fused QKV + SDPA attention** (replaces `nn.MultiheadAttention`, which
   does 3 separate QKV projections and a cuDNN/eager attention path):
   `bench_gpu.py` measures ~1.2x on the step. Small, mechanical, no risk.
2. **bf16 + no GradScaler**: bf16 has the same exponent range as fp32 so
   there's no scaling loss-precision concern; skips the fp16→bf16 conversion
   and scaler bookkeeping. Another ~1.2x.
3. **`torch.compile`**: another ~1.1–1.4x on top. Note: `torch.compile` on a
   model with `nn.MultiheadAttention` needs the SDPA swap first to compile
   cleanly; with the fused-QKV block it compiles to a single kernel per layer.
4. **fused AdamW** (`fused=True`): small but free.
5. **Batch size**: currently 1024; the GPU is saturated at 256 already, so
   larger batches don't add throughput, they just add VRAM. Not the lever here.

**Not worth doing:** the `torch.where`/`zeros_like`/`full_like` masking dance
is 0.2ms of a 330ms step — replacing it buys nothing. (If you ever want to
shrink memory, `masked_fill` is the cleaner idiom and is already what I use in
the bench.)

## The combined patch

`train.py` diff (apply when convenient; the running job is using the old code):

```diff
@@ model.py: replace Block.attn (nn.MultiheadAttention) with fused QKV + SDPA @@
-        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
+        self.qkv = nn.Linear(d, 3 * d)
+        self.proj = nn.Linear(d, d)
+        self.h = heads
+        self.dh = d // heads
...
-        n = self.ln1(x)
-        x = x + self.attn(n, n, n, need_weights=False)[0]
+        Bs, T, _ = x.shape
+        n = self.ln1(x)
+        q, k, v = self.qkv(n).split(d, dim=2)
+        q = q.view(Bs, T, self.h, self.dh).transpose(1, 2)
+        k = k.view(Bs, T, self.h, self.dh).transpose(1, 2)
+        v = v.view(Bs, T, self.h, self.dh).transpose(1, 2)
+        o = F.scaled_dot_product_attention(q, k, v)
+        o = o.transpose(1, 2).contiguous().view(Bs, T, d)
+        x = x + self.proj(o)

@@ train.py: bf16 autocast + fused AdamW + torch.compile @@
-        with torch.autocast("cuda", enabled=device.type == "cuda"):
+        with torch.autocast("cuda", dtype=torch.bfloat16):
...
-    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
-                            weight_decay=args.weight_decay)
+    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
+                            weight_decay=args.weight_decay, fused=True)
+    model = torch.compile(model)
```

(Note: `fused=True` requires CUDA-capable AdamW — it is here; and
`torch.compile` adds ~30–60s of compile time once at startup. If you keep
fp16+scaler instead of bf16, don't use `fused=True` without checking that the
optimizer and scaler interleave the same way.)

## Where to go from here

The cleanest 2x-ish win for this exact job is not a code change at all — it's
**the hardware**: the 4060 Laptop is a 60W part. On a desktop 4070+ or 4090
the same code would run 2–3x faster with zero changes. If this is a hobby
project, the pragmatic call is: apply patch items 1–4 (about 1.4–1.6x, ~1h40m
instead of 2h45m), and accept that further speedups need a bigger GPU or a
smaller model.

I did not modify `train.py` or restart the running job. The bench scripts
(`bench_data.py`, `bench_gpu.py`, `bench_batch.py`, `bench_stages.py`) are in
the repo for you to re-run when the GPU is free for clean numbers.
