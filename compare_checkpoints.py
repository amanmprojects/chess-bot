"""Score every saved checkpoint on the FULL val split and rank them.

The training-time eval uses a 2,048-position subsample (--eval-n), whose 1-sigma
noise is ~1.1pp -- enough that `best_acc` can pick a checkpoint by luck. This
re-scores each candidate over the whole val split so the pick is defensible.

Reports both metrics because they can disagree: top1 is what a move-playing bot
is judged on, CE also reflects calibration over the full distribution.

    python compare_checkpoints.py [--data-dir data]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from features import DTYPE, RECORD_BYTES, POLICY_SIZE
from model import ChessNet


def load_val(d):
    n = (d / "val.bin").stat().st_size // RECORD_BYTES
    recs = np.memmap(d / "val.bin", dtype=DTYPE, mode="r", shape=(n,))
    off = np.memmap(d / "legal_val_offsets.bin", dtype=np.uint64, mode="r")
    slots = np.memmap(d / "legal_val.bin", dtype=np.uint16, mode="r")
    return recs, off, slots, n


def score(path, recs, off, slots, n, device, batch=1024):
    ck = torch.load(path, map_location=device, weights_only=False)
    # Read the arch from the checkpoint so a d=320/L=8 model is comparable to a
    # d=256/L=7 one; older checkpoints have no metadata and fall back.
    net = ChessNet(d=int(ck.get("d", 256)), n_layers=int(ck.get("n_layers", 7)),
                   n_heads=int(ck.get("n_heads", 8))).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    ce_sum = acc_sum = vm_sum = total = 0
    with torch.no_grad():
        for s in range(0, n, batch):
            idx = np.arange(s, min(s + batch, n))
            rows = recs[idx]
            pieces = torch.from_numpy(np.asarray(rows["pieces"], dtype=np.int64)).to(device)
            aux = torch.from_numpy(np.asarray(rows["aux"], dtype=np.int64)).to(device)
            targets = torch.from_numpy(rows["policy"].astype(np.int64)).to(device)
            vals = torch.from_numpy(rows["value"].astype(np.float32)).to(device)
            mask = torch.zeros(len(idx), POLICY_SIZE, dtype=torch.bool)
            for r, i in enumerate(idx):
                lo, hi = int(off[i]), int(off[i + 1])
                mask[r, np.asarray(slots[lo:hi], dtype=np.int64)] = True
            mask = mask.to(device)
            logits, pred = net(pieces, aux)
            logits = logits.float()
            logits = logits + torch.where(mask, torch.zeros_like(logits),
                                          torch.full_like(logits, -1e9))
            ce_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            acc_sum += (logits.argmax(1) == targets).sum().item()
            vm_sum += F.mse_loss(pred, vals, reduction="sum").item()
            total += len(idx)
    return int(ck.get("step", -1)), ce_sum / total, acc_sum / total, vm_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs, off, slots, n = load_val(d)

    cands = sorted(set(list(d.glob("ckpt*.pt")) + list(d.glob("ckpt*.pt.final"))))
    if not cands:
        raise SystemExit("no checkpoints found")

    print(f"full val split: {n:,} positions   device {device}\n")
    print(f"{'checkpoint':<30}{'step':>8}{'CE':>10}{'top1':>9}{'val_mse':>10}")
    rows = []
    for p in cands:
        try:
            st, ce, acc, vm = score(p, recs, off, slots, n, device)
        except Exception as exc:
            print(f"{p.name:<30}  FAILED: {type(exc).__name__}: {exc}")
            continue
        rows.append((p.name, st, ce, acc, vm))
        print(f"{p.name:<30}{st:>8,}{ce:>10.4f}{acc:>9.4f}{vm:>10.4f}")

    if len(rows) > 1:
        se = math.sqrt(0.43 * 0.57 / n)
        # Several files can hold the same weights (ckpt.pt is a copy of the last
        # saved best). Rank distinct models so the margin is meaningful.
        uniq = {}
        for name, st, ce, acc, vm in rows:
            key = (st, round(ce, 6), round(acc, 6))
            uniq.setdefault(key, []).append(name)
        distinct = [(names, k[0], k[1], k[2]) for k, names in uniq.items()]
        best_acc = max(distinct, key=lambda r: r[3])
        best_ce = min(distinct, key=lambda r: r[2])
        print(f"\n1-sigma on top1 at this sample size: {se*100:.2f}pp")
        print(f"distinct models: {len(distinct)} "
              f"({', '.join('=' .join(n) for n, *_ in distinct)})")
        print(f"best top1: {'/'.join(best_acc[0])} ({best_acc[3]:.4f})")
        print(f"best CE  : {'/'.join(best_ce[0])} ({best_ce[2]:.4f})")
        if best_acc[0] == best_ce[0]:
            print(f"\n=> RECOMMEND {'/'.join(best_acc[0])} (wins on both metrics)")
        elif len(distinct) > 1:
            runner = sorted((r[3] for r in distinct), reverse=True)[1]
            print(f"\n=> metrics disagree. For a move-playing bot prefer top1: "
                  f"{'/'.join(best_acc[0])}")
            print(f"   margin over next distinct model "
                  f"{(best_acc[3]-runner)*100:+.2f}pp = "
                  f"{abs(best_acc[3]-runner)/se:.1f} sigma"
                  f"{'  (within noise -- either is defensible)' if abs(best_acc[3]-runner)/se < 2 else ''}")


if __name__ == "__main__":
    main()
