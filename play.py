"""Evaluate the trained model: holdout accuracy, legal-move rate, and games vs Stockfish.

Usage:
    python play.py --ckpt data/ckpt.pt --mode eval
    python play.py --ckpt data/ckpt.pt --mode stockfish --games 20 --depth 8
"""

import argparse
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

from features import (DTYPE, RECORD_BYTES, POLICY_SIZE, board_to_features,
                      features_to_board, move_to_slot, policy_mask)
from model import ChessNet

STOCKFISH = "stockfish"


def load_model(path, device, d, n_layers):
    ckpt = torch.load(path, map_location=device)
    model = ChessNet(d=d, n_layers=n_layers).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def pick_move(model, board, device, temperature):
    pieces, aux = board_to_features(board)
    p = torch.from_numpy(pieces).long().unsqueeze(0).to(device)
    a = torch.from_numpy(aux).long().unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
        logits, value = model(p, a)
    mask = torch.from_numpy(policy_mask(board)).to(device)
    # Must stay float: -1e9 overflows float16 (max ~65504), which torch.where
    # raises on rather than saturating.
    masked = logits[0].float()
    masked = masked + torch.where(mask, torch.zeros_like(masked),
                                  torch.full_like(masked, -1e9))
    probs = F.softmax(masked / max(temperature, 1e-4), dim=0)
    slot = probs.argmax().item() if temperature <= 0 else \
        torch.multinomial(probs, 1).item()
    for move in board.legal_moves:
        if move_to_slot(board, move) == slot:
            return move, value.item()
    raise RuntimeError(f"slot {slot} not found among legal moves in {board.fen()}")


def eval_mode(model, val_path, n, batch_size, device):
    size = Path(val_path).stat().st_size
    n_val = size // RECORD_BYTES
    assert n_val * RECORD_BYTES == size
    mmap = np.memmap(val_path, dtype=DTYPE, mode="r", shape=(n_val,))
    rng = np.random.default_rng(0)
    idx = rng.choice(n_val, size=min(n, n_val), replace=False)
    acc = legal_ok = total = 0
    for i in idx:
        pieces, aux = mmap[i]["pieces"], mmap[i]["aux"]
        board = features_to_board(pieces, aux)
        target = int(mmap[i]["policy"])
        p = torch.from_numpy(pieces).long().unsqueeze(0).to(device)
        a = torch.from_numpy(aux).long().unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(p, a)
        mask = torch.from_numpy(policy_mask(board)).to(device)
        # Cast out of autocast's float16 before masking: -1e9 overflows half
        # (max ~65504) and torch.where raises rather than saturating. Same fix
        # as train.py's evaluate().
        masked = logits[0].float()
        masked = masked + torch.where(mask, torch.zeros_like(masked),
                                      torch.full_like(masked, -1e9))
        pred = masked.argmax().item()
        acc += int(pred == target)
        legal_ok += int(mask[pred])
        total += 1
    print(f"[eval] n={total} top1 {acc/total:.4f} legal {legal_ok/total:.4f}")


def stockfish_mode(model, engine_path, games, depth, movetime, temperature,
                   device, seed, start_fen=None, skill=None):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    if skill is not None:
        engine.configure({"Skill Level": skill})
    rng = random.Random(seed)
    results = []
    for g in range(games):
        board = chess.Board(start_fen) if start_fen else chess.Board()
        moves = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                if g % 2:
                    result = engine.play(board, chess.engine.Limit(depth=depth,
                                                                   time=movetime))
                    board.push(result.move)
                else:
                    move, _ = pick_move(model, board, device, temperature)
                    board.push(move)
            else:
                if g % 2:
                    move, _ = pick_move(model, board, device, temperature)
                    board.push(move)
                else:
                    result = engine.play(board, chess.engine.Limit(depth=depth,
                                                                   time=movetime))
                    board.push(result.move)
            moves.append(board.peek())
        r = board.result()
        results.append(r)
        outcome = {"1-0": "W" if g % 2 == 0 else "L",
                   "0-1": "L" if g % 2 == 0 else "W"}.get(r, "D")
        print(f"game {g+1}/{games} ({'model=W' if g % 2 == 0 else 'model=B'}) "
              f"{r} -> {outcome}  plies {len(moves)}", flush=True)
    engine.quit()
    score = 0
    for g, r in enumerate(results):
        if r == "1/2-1/2":
            score += 0.5
        elif r == "1-0" and g % 2 == 0 or r == "0-1" and g % 2 == 1:
            score += 1.0
    print(f"[stockfish] {games} games vs {engine_path} (depth {depth}): "
          f"model score {score:.1f}/{games} = {score/max(games,1)*100:.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpt.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--mode", choices=["eval", "stockfish"], default="eval")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--movetime", type=float, default=None)
    ap.add_argument("--skill", type=int, default=None,
                    help="Stockfish Skill Level 0-20 (roughly human Elo: 1 ~ 1000)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ckpt = load_model(args.ckpt, device, args.d, args.n_layers)
    step_str = ckpt.get('step', '?')
    acc_val = ckpt.get('best_acc', None)
    acc_str = f"{acc_val:.4f}" if acc_val is not None else "?"
    print(f"loaded {args.ckpt} (step {step_str}, best acc {acc_str})")

    if args.mode == "eval":
        eval_mode(model, str(Path(args.data_dir) / "val.bin"),
                  args.eval_n, 64, device)
    else:
        stockfish_mode(model, STOCKFISH, args.games, args.depth, args.movetime,
                       args.temperature, device, args.seed, skill=args.skill)
