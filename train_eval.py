"""Train the policy+value net with Stockfish eval targets instead of game results.

The shipped model regresses its value head on the game outcome (+1/0/-1), which
says nothing about the position: a winning position that was thrown away 30 moves
later is labelled -1. Measured MSE for that head is 0.75, roughly what a constant
predictor scores on this data.

This variant swaps that target for a real engine score. train.evl holds int16
centipawns from the side-to-move's view, aligned index-for-index with train.bin
(produced by modal/eval_app.py, merged by modal/merge_evals.py). The policy loss
is untouched -- the point is not to change what the net imitates, only to give
the value head something worth ranking moves with.

Loss = policy CE + value_weight * Huber(pred_cp, target_cp), where pred_cp is
2000*tanh(raw) so the head's existing [-1,1] output is reinterpreted as
centipawns. Huber rather than MSE because +-2000cp clamping leaves a long tail
and MSE would spend the gradient budget on the clamped positions.

Usage:
    python train_eval.py --data-dir data --eval-file data/train.evl \
        --epochs 12 --out data/ckpt_eval.pt
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
from train import load_memmap, load_legal, mask_batch_from_legal

CP_SCALE = 2000.0


def load_evals(path, n_records):
    """int16 centipawns -> float32, verified against the record count."""
    size = Path(path).stat().st_size
    n = size // 2
    if n != n_records:
        raise SystemExit(f"{path}: {n:,} evals but train.bin has {n_records:,} "
                         f"records. Refusing to train on misaligned targets.")
    return np.array(np.memmap(path, dtype=np.int16, mode="r"))


@torch.no_grad()
def evaluate(model, val_mmap, slots, off, idx, batch_size, device, val_evals):
    ce_sum = acc_sum = hub_sum = corr_num = corr_den = 0.0
    n_seen = 0
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        rows = val_mmap[batch]
        pieces = torch.from_numpy(np.asarray(rows["pieces"], dtype=np.int64)).to(device)
        aux = torch.from_numpy(np.asarray(rows["aux"], dtype=np.int64)).to(device)
        targets = torch.from_numpy(rows["policy"].astype(np.int64)).to(device)
        target_cp = torch.from_numpy(
            val_evals[batch].astype(np.float32)).to(device)
        masks = mask_batch_from_legal(slots, off, batch, device)
        logits, pred = model(pieces, aux)
        logits = logits.float()
        masked = logits + torch.where(masks, torch.zeros_like(logits), -1e9)
        ce = F.cross_entropy(masked, targets)
        acc = (masked.argmax(1) == targets).float().mean()
        pred_cp = torch.tanh(pred) * CP_SCALE
        hub = F.smooth_l1_loss(pred_cp, target_cp, beta=200.0)
        n_seen += pieces.shape[0]
        ce_sum += ce.item() * pieces.shape[0]
        acc_sum += acc.item() * pieces.shape[0]
        hub_sum += hub.item() * pieces.shape[0]
        # Signed-agreement rate: the fraction of positions where the head calls
        # the winner the same way Stockfish does. Top1 alone says nothing about
        # the value head; this is the number that has to move for search to work.
        agree = ((pred_cp > 0) == (target_cp > 0)).float().mean()
        corr_num += agree.item() * pieces.shape[0]
        corr_den += pieces.shape[0]
    return (ce_sum / n_seen, acc_sum / n_seen, hub_sum / n_seen,
            corr_num / corr_den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval-file", default="data/train.evl",
                    help="int16 centipawns, index-aligned with train.bin")
    ap.add_argument("--val-eval-file", default="",
                    help="Optional evals for val.bin. Falls back to --eval-file "
                         "sliced at the train/val boundary only if absent.")
    ap.add_argument("--out", default="data/ckpt_eval.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--value-weight", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_mmap, n_train = load_memmap(Path(args.data_dir) / "train.bin")
    val_mmap, n_val = load_memmap(Path(args.data_dir) / "val.bin")
    train_slots, train_off, n_legal = load_legal(
        Path(args.data_dir) / "legal.bin",
        Path(args.data_dir) / "legal_offsets.bin")
    val_slots, val_off, _ = load_legal(Path(args.data_dir) / "legal_val.bin",
                                       Path(args.data_dir) / "legal_val_offsets.bin")
    assert n_legal == n_train

    train_evals = load_evals(args.eval_file, n_train)
    if args.val_eval_file:
        val_evals = load_evals(args.val_eval_file, n_val)
    else:
        raise SystemExit("--val-eval-file is required; eval the val split too "
                         "(modal run eval_app.py --src val.bin).")

    print(f"train {n_train:,} records  val {n_val:,}  device {device}")
    print(f"[evals] train mean {train_evals.mean():.1f} cp  "
          f"std {train_evals.std():.1f}  "
          f"clamped {(np.abs(train_evals) >= CP_SCALE).sum():,} "
          f"({(np.abs(train_evals) >= CP_SCALE).mean()*100:.2f}%)")

    model = ChessNet(d=args.d, n_layers=args.n_layers,
                     n_heads=args.n_heads).to(device)
    print(f"model: {model.num_params()/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = (n_train - n_train % args.batch_size) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    scaler = torch.amp.GradScaler("cuda")
    step, best_acc, best_agree = 0, 0.0, 0.0
    if args.resume and Path(args.out).exists():
        ck = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        best_acc = float(ck.get("best_acc", 0.0))
        best_agree = float(ck.get("best_agree", 0.0))
        print(f"resumed from {args.out} at step {step:,}")

    t0 = time.time()
    run_steps = 0
    rng_eval = np.random.default_rng(0)
    idx = rng_eval.choice(n_val, size=min(args.eval_n, n_val), replace=False)
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
            target_cp = torch.from_numpy(
                train_evals[batch].astype(np.float32)).to(device)
            masks = mask_batch_from_legal(train_slots, train_off, batch, device)

            lr = args.lr * _lr_scale(step, args.warmup_steps, total_steps,
                                     args.min_lr / args.lr)
            opt.param_groups[0]["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits, pred = model(pieces, aux)
                logits = logits.float()
                masked = logits + torch.where(masks, torch.zeros_like(logits), -1e9)
                ce = F.cross_entropy(masked, targets)
                hub = F.smooth_l1_loss(torch.tanh(pred) * CP_SCALE, target_cp,
                                       beta=200.0)
                loss = ce + args.value_weight * hub
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            run_steps += 1
            if run_steps % 50 == 0:
                rate = run_steps * args.batch_size / (time.time() - t0)
                print(f"step {step:>6} loss {loss.item():.4f} ce {ce.item():.4f} "
                      f"vhub {hub.item():.2f} lr {lr:.2e} "
                      f"samples/s {rate:.0f} elapsed {(time.time()-t0)/60:.1f}min",
                      flush=True)

            if step % args.eval_every == 0:
                vce, vacc, vhub, vagree = evaluate(
                    model, val_mmap, val_slots, val_off, idx, args.batch_size,
                    device, val_evals)
                print(f"  [eval] step {step} ce {vce:.4f} top1 {vacc:.4f} "
                      f"vhub {vhub:.2f} agree {vagree:.4f}", flush=True)
                # Gate on the value head agreeing with Stockfish, with top1 as a
                # tiebreak: a checkpoint that can rank positions is worth more
                # than one that matches humans slightly more often.
                if vagree > best_agree or (vagree == best_agree and vacc > best_acc):
                    best_agree, best_acc = vagree, vacc
                    torch.save({"model": model.state_dict(),
                                "opt": opt.state_dict(), "step": step,
                                "best_acc": best_acc, "best_agree": best_agree,
                                "d": args.d, "n_layers": args.n_layers,
                                "n_heads": args.n_heads,
                                "eval_scale": CP_SCALE}, args.out)
                    print(f"  [ckpt] saved {args.out} "
                          f"(agree {best_agree:.4f} top1 {best_acc:.4f})", flush=True)

            if stop_at and time.time() >= stop_at:
                print(f"[deadline] stopping at step {step:,}", flush=True)
                out_of_time = True
                break

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best_acc": best_acc, "best_agree": best_agree,
                "d": args.d, "n_layers": args.n_layers, "n_heads": args.n_heads,
                "eval_scale": CP_SCALE}, args.out + ".final")
    try:
        vce, vacc, vhub, vagree = evaluate(model, val_mmap, val_slots, val_off,
                                           idx, args.batch_size, device, val_evals)
        noise = math.sqrt(max(vagree, 1e-6) * (1 - vagree) / max(len(idx), 1))
        print(f"final ce {vce:.4f} top1 {vacc:.4f} vhub {vhub:.2f} "
              f"agree {vagree:.4f} best {best_agree:.4f}")
        if vagree > best_agree + noise:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_acc": vacc, "best_agree": vagree,
                        "d": args.d, "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "eval_scale": CP_SCALE}, args.out)
    except Exception as exc:
        print(f"[warn] final eval failed: {exc}", flush=True)
    print("[done]", flush=True)


def _lr_scale(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    if step >= total:
        return min_ratio
    prog = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


if __name__ == "__main__":
    main()