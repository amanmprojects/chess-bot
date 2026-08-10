"""CPU-side benchmark of the train.py batch pipeline (no GPU, safe to run
alongside a live training job). Times each stage of one batch, averaged."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features import DTYPE, RECORD_BYTES, POLICY_SIZE

DATA = ROOT / "data"
B = 1024
ITERS = 30


def load_memmap(path):
    size = Path(path).stat().st_size
    n = size // RECORD_BYTES
    return np.array(np.memmap(path, dtype=DTYPE, mode="r", shape=(n,))), n


def main():
    t = time.perf_counter()
    train, n = load_memmap(DATA / "train.bin")
    off = np.array(np.memmap(DATA / "legal_offsets.bin", dtype=np.uint64, mode="r"))
    slots = np.array(np.memmap(DATA / "legal.bin", dtype=np.uint16, mode="r"))
    print(f"loaded {n:,} records in {time.perf_counter()-t:.1f}s")
    print(f"avg legal moves/pos {slots.size/n:.1f}")

    perm = np.random.default_rng(0).permutation(n)
    stages = {k: 0.0 for k in
              ("gather", "pieces", "aux", "targets", "vals", "mask_numpy")}

    for i in range(ITERS):
        batch = perm[i * B:(i + 1) * B]

        t = time.perf_counter()
        rows = train[batch]
        stages["gather"] += time.perf_counter() - t

        t = time.perf_counter()
        np.asarray(rows["pieces"], dtype=np.int64)
        stages["pieces"] += time.perf_counter() - t

        t = time.perf_counter()
        np.asarray(rows["aux"], dtype=np.int64)
        stages["aux"] += time.perf_counter() - t

        t = time.perf_counter()
        rows["policy"].astype(np.int64)
        stages["targets"] += time.perf_counter() - t

        t = time.perf_counter()
        rows["value"].astype(np.float32)
        stages["vals"] += time.perf_counter() - t

        t = time.perf_counter()
        lo, hi = off[batch], off[batch + 1]
        lengths = (hi - lo).astype(np.int64)
        flat = np.concatenate([slots[lo[j]:hi[j]] for j in range(B)]).astype(np.int64)
        np.repeat(np.arange(B), lengths)
        stages["mask_numpy"] += time.perf_counter() - t

    total = sum(stages.values()) / ITERS * 1000
    print(f"\nper batch of {B} (avg over {ITERS}):")
    for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v/ITERS*1000:7.2f} ms")
    print(f"  {'TOTAL':<12} {total:7.2f} ms   -> {B/(total/1000):.0f} rec/s ceiling")


if __name__ == "__main__":
    main()
