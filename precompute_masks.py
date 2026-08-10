"""Precompute legal-move slot lists for every record, compactly.

The records already encode the full board state, so the mask computation that
made training CPU-bound can be done once here instead of every epoch:

    train.bin      -> legal.bin          (uint16 slots, concatenated)
                       legal_offsets.bin (uint64 cumulative boundaries)
    val.bin        -> legal_val.bin, legal_val_offsets.bin

At train time a batch's mask is rebuilt on GPU by scatter from these lists.

Usage:
    python precompute_masks.py --data-dir data
"""

import argparse
import time
from pathlib import Path

import numpy as np

from features import DTYPE, RECORD_BYTES, features_to_board, policy_mask


def process(src_path, legal_path, offsets_path, log_every=250_000):
    n = src_path.stat().st_size // RECORD_BYTES
    assert n * RECORD_BYTES == src_path.stat().st_size
    mmap = np.memmap(src_path, dtype=DTYPE, mode="r", shape=(n,))
    buf = bytearray()
    offsets = np.empty(n + 1, dtype=np.uint64)
    offsets[0] = 0
    t0 = time.time()
    for i in range(n):
        board = features_to_board(mmap[i]["pieces"], mmap[i]["aux"])
        slots = np.nonzero(policy_mask(board))[0].astype(np.uint16)
        buf += slots.tobytes()
        offsets[i + 1] = offsets[i] + len(slots)
        if i and i % log_every == 0:
            rate = i / (time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] {i:,}/{n:,} "
                  f"({rate:.0f}/s, {100*i/n:.0f}%)", flush=True)
    with open(legal_path, "wb") as fh:
        fh.write(buf)
    offsets.tofile(offsets_path)
    avg = offsets[n] / max(n, 1)
    print(f"[done] {n:,} records -> {legal_path} "
          f"({offsets[n]/1e6:.1f}M slots, avg {avg:.1f} legal moves)  "
          f"{offsets_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = Path(args.data_dir)
    process(d / "train.bin", d / "legal.bin", d / "legal_offsets.bin")
    process(d / "val.bin", d / "legal_val.bin", d / "legal_val_offsets.bin")


if __name__ == "__main__":
    main()
