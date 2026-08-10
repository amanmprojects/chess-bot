"""Independent validation of prepared records + precomputed masks.

Re-derives everything from the stored bytes and checks the invariants training
depends on. Run after prepare.py + precompute_masks.py, before training.

    python validate_data.py --data-dir data
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from features import (DTYPE, RECORD_BYTES, POLICY_SIZE, features_to_board,
                      policy_mask)


def load(path):
    n = Path(path).stat().st_size // RECORD_BYTES
    assert n * RECORD_BYTES == Path(path).stat().st_size, f"{path} ragged"
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(n,)), n


def load_legal(slots_path, offsets_path):
    off = np.memmap(offsets_path, dtype=np.uint64, mode="r")
    slots = np.memmap(slots_path, dtype=np.uint16, mode="r")
    return slots, off, len(off) - 1


def check(name, records_path, slots_path, offsets_path, sample, rng):
    print(f"\n=== {name} ===")
    recs, n = load(records_path)
    slots, off, n_legal = load_legal(slots_path, offsets_path)
    ok = True

    if n_legal != n:
        print(f"  FAIL mask count {n_legal:,} != record count {n:,}")
        ok = False
    if int(off[-1]) != slots.size:
        print(f"  FAIL offsets tail {int(off[-1])} != slots size {slots.size}")
        ok = False
    print(f"  records {n:,}  slots {slots.size:,}  avg legal {slots.size/max(n,1):.1f}")

    pol = np.asarray(recs["policy"], dtype=np.int64)
    if pol.min() < 0 or pol.max() >= POLICY_SIZE:
        print(f"  FAIL policy out of range [{pol.min()}, {pol.max()}]")
        ok = False
    frm, lab = pol // 73, pol % 73
    n_frm, n_lab = len(np.unique(frm)), len(np.unique(lab))
    print(f"  distinct from-squares {n_frm}/64   distinct labels {n_lab}/73")
    # Guard against the historical square-blind encoding bug, where every target
    # collapsed onto a single from-square. Scale the bar to the record count so a
    # legitimately tiny split cannot trip it.
    min_frm = min(40, max(2, n // 4))
    if n_frm < min_frm:
        print(f"  FAIL only {n_frm} distinct from-squares over {n:,} records "
              f"(expected >= {min_frm}) -- targets look square-blind")
        ok = False

    val = np.asarray(recs["value"], dtype=np.int64)
    counts = collections.Counter(val.tolist())
    if not set(counts) <= {-1, 0, 1}:
        print(f"  FAIL value outside -1/0/1: {sorted(set(counts))}")
        ok = False
    total = sum(counts.values())
    print(f"  value  win {counts.get(1,0)/total:.3f}  "
          f"draw {counts.get(0,0)/total:.3f}  loss {counts.get(-1,0)/total:.3f}")

    # The invariant training relies on: every target is inside its own stored mask,
    # and the stored mask equals the mask recomputed from the stored features.
    idx = rng.choice(n, size=min(sample, n), replace=False)
    bad_target = bad_mask = 0
    for i in idx:
        i = int(i)
        lo, hi = int(off[i]), int(off[i + 1])
        stored = np.zeros(POLICY_SIZE, dtype=bool)
        stored[np.asarray(slots[lo:hi], dtype=np.int64)] = True
        if not stored[pol[i]]:
            bad_target += 1
        board = features_to_board(recs[i]["pieces"], recs[i]["aux"])
        if not np.array_equal(stored, policy_mask(board)):
            bad_mask += 1
    print(f"  sampled {len(idx):,}: target-in-mask failures {bad_target}, "
          f"mask-mismatch failures {bad_mask}")
    if bad_target or bad_mask:
        ok = False

    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    ok = check("train", d / "train.bin", d / "legal.bin",
               d / "legal_offsets.bin", args.sample, rng)
    ok &= check("val", d / "val.bin", d / "legal_val.bin",
                d / "legal_val_offsets.bin", args.sample, rng)
    print("\n" + ("ALL CHECKS PASSED" if ok else "VALIDATION FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
