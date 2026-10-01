"""Train the policy+value chess transformer on the prepared record bins.

Usage:
    python train.py --data-dir data --epochs 20 --out data/ckpt.pt
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from features import DTYPE, RECORD_BYTES, POLICY_SIZE
from model import ChessNet


def load_memmap(path):
    size = Path(path).stat().st_size
    n = size // RECORD_BYTES
    assert n * RECORD_BYTES == size, f"{path}: size {size} not divisible by {RECORD_BYTES}"
    return np.array(np.memmap(path, dtype=DTYPE, mode="r", shape=(n,))), n


def load_legal(slots_path, offsets_path):
    off = np.array(np.memmap(offsets_path, dtype=np.uint64, mode="r"))
    n = len(off) - 1
    slots = np.array(np.memmap(slots_path, dtype=np.uint16, mode="r"))
    assert slots.size == int(off[-1])
    return slots, off, n


def mask_batch_from_legal(slots, off, indices, device):
    """Boolean legal-move masks (B, 4672) on GPU, scattered from the slot lists."""
    B = len(indices)
    lo = off[indices]
    hi = off[indices + 1]
    lengths = (hi - lo).astype(np.int64)
    flat = np.concatenate([slots[lo[i]:hi[i]] for i in range(B)]).astype(np.int64)
    rows = np.repeat(np.arange(B), lengths)
    masks = torch.zeros(B, POLICY_SIZE, dtype=torch.bool, device=device)
    masks[torch.from_numpy(rows).to(device),
          torch.from_numpy(flat).to(device)] = True
    return masks


@torch.no_grad()
def evaluate(model, val_mmap, slots, off, idx, batch_size, device):
    ce_sum = acc_sum = val_sum = n_seen = 0
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        rows = val_mmap[batch]
        pieces = torch.from_numpy(np.asarray(rows["pieces"], dtype=np.int64)).to(device)
        aux = torch.from_numpy(np.asarray(rows["aux"], dtype=np.int64)).to(device)
        targets = torch.from_numpy(rows["policy"].astype(np.int64)).to(device)
        vals = torch.from_numpy(rows["value"].astype(np.float32)).to(device)
        masks = mask_batch_from_legal(slots, off, batch, device)
        logits, pred = model(pieces, aux)
        logits = logits.float()
        masked = logits + torch.where(masks, torch.zeros_like(logits), -1e9)
        ce = F.cross_entropy(masked, targets)
        acc = (masked.argmax(1) == targets).float().mean()
        n_seen += pieces.shape[0]
        ce_sum += ce.item() * pieces.shape[0]
        acc_sum += acc.item() * pieces.shape[0]
        val_sum += F.mse_loss(pred, vals).item() * pieces.shape[0]
    return ce_sum / n_seen, acc_sum / n_seen, val_sum / n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/ckpt.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="Unix timestamp. Stop training gracefully before this "
                         "time, still saving weights and running the final "
                         "eval. 0 disables. Survives --resume because it is an "
                         "absolute time, not a per-attempt budget.")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the checkpoint in --out if present")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_mmap, n_train = load_memmap(Path(args.data_dir) / "train.bin")
    val_mmap, n_val = load_memmap(Path(args.data_dir) / "val.bin")
    train_slots, train_off, n_legal = load_legal(Path(args.data_dir) / "legal.bin",
                                                 Path(args.data_dir) / "legal_offsets.bin")
    val_slots, val_off, _ = load_legal(Path(args.data_dir) / "legal_val.bin",
                                       Path(args.data_dir) / "legal_val_offsets.bin")
    assert n_legal == n_train
    print(f"train {n_train:,} records  val {n_val:,}  device {device}")

    model = ChessNet(d=args.d, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    print(f"model: {model.num_params()/1e6:.2f}M params "
          f"({model.num_params(True)/1e6:.2f}M trainable)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = (n_train - n_train % args.batch_size) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    best_acc = 0.0
    if args.resume and Path(args.out).exists():
        ck = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        best_acc = float(ck["best_acc"])
        print(f"resumed from {args.out} at step {step:,} (best acc {best_acc:.4f})")
    t0 = time.time()
    run_steps = 0

    rng_eval = np.random.default_rng(0)
    idx = rng_eval.choice(n_val, size=min(args.eval_n, n_val), replace=False)

    # Reserve time for the final save + eval so hitting the deadline still ends
    # through the normal [done] path instead of being killed mid-write.
    stop_at = (args.deadline - 180) if args.deadline else 0.0
    out_of_time = False

    start_epoch = step // steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        if out_of_time:
            break
        perm = np.random.default_rng(args.seed + epoch).permutation(n_train)
        first_local = step % steps_per_epoch if epoch == start_epoch else 0
        for li in range(first_local, steps_per_epoch):
            batch = perm[li * args.batch_size:(li + 1) * args.batch_size]
            rows = train_mmap[batch]
            pieces = torch.from_numpy(np.asarray(rows["pieces"], dtype=np.int64)).to(device)
            aux = torch.from_numpy(np.asarray(rows["aux"], dtype=np.int64)).to(device)
            targets = torch.from_numpy(rows["policy"].astype(np.int64)).to(device)
            vals = torch.from_numpy(rows["value"].astype(np.float32)).to(device)
            masks = mask_batch_from_legal(train_slots, train_off, batch, device)

            opt.param_groups[0]["lr"] = args.lr * _lr_scale(
                step, args.warmup_steps, total_steps, args.min_lr / args.lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits, pred = model(pieces, aux)
                logits = logits.float()
                masked = logits + torch.where(masks, torch.zeros_like(logits), -1e9)
                ce = F.cross_entropy(masked, targets)
                loss = ce + args.value_weight * F.mse_loss(pred, vals)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            run_steps += 1
            if run_steps % 50 == 0:
                lr = opt.param_groups[0]["lr"]
                rate = run_steps * args.batch_size / (time.time() - t0)
                print(f"step {step:>6} loss {loss.item():.4f} ce {ce.item():.4f} "
                      f"lr {lr:.2e} tok/s {rate:.0f} "
                      f"elapsed {(time.time()-t0)/60:.1f}min", flush=True)

            if step % args.eval_every == 0:
                vce, vacc, vv = evaluate(model, val_mmap, val_slots, val_off,
                                         idx, args.batch_size, device)
                lr = opt.param_groups[0]["lr"]
                print(f"  [eval] step {step} ce {vce:.4f} top1 {vacc:.4f} "
                      f"value_mse {vv:.4f} lr {lr:.2e}", flush=True)
                if vacc > best_acc:
                    best_acc = vacc
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "best_acc": best_acc,
                                "d": args.d, "n_layers": args.n_layers,
                                "n_heads": args.n_heads}, args.out)
                    print(f"  [ckpt] saved {args.out} (best acc {best_acc:.4f})",
                          flush=True)

            if stop_at and time.time() >= stop_at:
                print(f"[deadline] stopping at step {step:,} "
                      f"(epoch {epoch}/{args.epochs}) to save and eval before "
                      f"the wall-clock deadline", flush=True)
                out_of_time = True
                break

    # Save the final weights BEFORE evaluating: a crash in evaluate() must never
    # discard a completed training run.
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best_acc": best_acc,
                "d": args.d, "n_layers": args.n_layers,
                "n_heads": args.n_heads}, args.out + ".final")
    try:
        vce, vacc, vv = evaluate(model, val_mmap, val_slots, val_off, idx,
                                 args.batch_size, device)
        summary = (f"final ce {vce:.4f} top1 {vacc:.4f} value_mse {vv:.4f} "
                   f"best {best_acc:.4f}")
        # Both vacc and best_acc come from the same --eval-n subsample, so a
        # small win is indistinguishable from noise. Require more than 1 sigma
        # before replacing a checkpoint that already proved itself in training.
        noise = math.sqrt(max(vacc, 1e-6) * (1 - vacc) / max(len(idx), 1))
        if vacc > best_acc + noise:
            best_acc = vacc
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_acc": best_acc,
                        "d": args.d, "n_layers": args.n_layers,
                        "n_heads": args.n_heads}, args.out)
            print(f"  [ckpt] saved {args.out} (best acc {best_acc:.4f})", flush=True)
    except Exception as exc:
        summary = f"final eval failed ({type(exc).__name__}: {exc}) best {best_acc:.4f}"
    print(f"[done] {summary} (final model saved to {args.out}.final)", flush=True)


def _lr_scale(step, warmup, total, min_ratio):
    if step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * prog))


if __name__ == "__main__":
    main()
