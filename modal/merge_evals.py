"""Merge the per-shard Stockfish eval outputs back into one aligned .evl file.

eval_app.py shards by stride (worker i owns indices i, i+S, i+2S, ...) and writes
its own file, so reassembly is an interleave, not a concatenate:

    full[shard::num_shards] = shard_file[shard]

Output is int16 centipawns from the side-to-move's view, same index space as
train.bin, so training can memmap both and index them together.
"""

import argparse
from pathlib import Path

import numpy as np


def merge(shard_dir: Path, n_records: int, num_shards: int, out: Path,
          prefix: str = "train.evl"):
    out_array = np.zeros(n_records, dtype=np.int16)
    seen = 0
    for shard in range(num_shards):
        path = shard_dir / f"{prefix}.{shard:03d}"
        data = np.fromfile(path, dtype=np.int16)
        expected = len(range(shard, n_records, num_shards))
        if len(data) != expected:
            raise SystemExit(
                f"{path}: got {len(data)} evals, expected {expected}. "
                f"A shard did not finish -- do not train on a partial eval.")
        out_array[shard::num_shards] = data
        seen += len(data)
    if seen != n_records:
        raise SystemExit(f"merged {seen} evals, expected {n_records}")
    out_array.tofile(out)
    print(f"[merge] {seen:,} evals -> {out} "
          f"(mean {out_array.mean():.1f} cp, std {out_array.std():.1f}, "
          f"clamped {(np.abs(out_array) >= 2000).sum():,})")
    return out_array


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--n-records", type=int, default=2_587_000)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="train.evl",
                    help="shard filename prefix, e.g. train.evl or val.evl")
    args = ap.parse_args()
    merge(Path(args.shard_dir), args.n_records, args.shards, Path(args.out),
          args.prefix)