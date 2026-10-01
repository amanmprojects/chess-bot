"""A/B the move-selection strategies against Stockfish, measuring blunders.

This is the measurement that decides whether search.py is worth shipping. It
plays the bot under each strategy from the same openings and counts, per game:

  * illegal moves / forfeits
  * blunders: the side to move drops >= 200cp of material-weighted advantage
    between the position before and after its own move, by Stockfish's eval.

Blunder rate is the honest version of the user's complaint. A score against a
fixed opponent conflates blunders with ordinary weak play; a blunder count does
not.

Usage:
    python bench_blunders.py --ckpt data/ckpt.pt --games 12 --modes policy guard guard+bon guard+ab
"""

import argparse
import time

import chess
import chess.engine
import torch

from features import move_to_slot, policy_mask
from model import ChessNet
import search as S

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Score at depth 8, capped so a forced mate does not read as a 30000cp blunder.
BLUNDER_CP = 200
MATE_CAP = 10000


def score_cp(engine, board, depth, pov):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].pov(pov).score(mate_score=MATE_CAP)


def material(board, color):
    return sum(PIECE_VALUE[p.piece_type] for p in board.piece_map().values()
               if p.color == color)


def play_game(engine, net, device, mode, seed, max_plies=200, depth=6):
    """One game vs Stockfish. Returns (result_for_model, blunders, forfeits, plies)."""
    import random
    rng = random.Random(seed)
    board = chess.Board()
    model_white = (seed % 2 == 0)
    blunders = forfeits = 0

    for ply in range(max_plies):
        if board.is_game_over():
            break
        if board.turn == (chess.WHITE if model_white else chess.BLACK):
            pov = board.turn
            before = score_cp(engine, board, depth, pov)
            try:
                move, _ = S.search_move(board, net, device, mode=mode, depth=2, n=8)
            except Exception:
                forfeits += 1
                break
            board.push(move)
            after = score_cp(engine, board, depth, pov)
            # Centipawns from our side before vs after our own move. A large
            # negative swing that is not a sacrifice we knowingly accepted is a
            # blunder; require it to exceed the threshold to ignore engine noise.
            drop = before - after
            if drop >= BLUNDER_CP:
                blunders += 1
        else:
            board.push(engine.play(board, chess.engine.Limit(depth=depth)).move)

    result = board.result()
    if result == "1/2-1/2":
        outcome = 0.5
    elif (result == "1-0") == model_white:
        outcome = 1.0
    else:
        outcome = 0.0
    return outcome, blunders, forfeits, ply + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpt.pt")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--modes", default="policy,guard,guard+bon,guard+ab")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--stockfish-skill", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = ChessNet(d=int(ck.get("d", 256)), n_layers=int(ck.get("n_layers", 7)),
                   n_heads=int(ck.get("n_heads", 8))).to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    engine = chess.engine.SimpleEngine.popen_uci("/run/media/aman/arch/home/aman/.local/bin/stockfish")
    if args.stockfish_skill is not None:
        engine.configure({"Skill Level": args.stockfish_skill})

    modes = args.modes.split(",")
    print(f"{args.games} games/opponent, depth {args.depth}, "
          f"{args.max_plies}-ply cap, ckpt step {ck.get('step')}\n")
    header = f"{'mode':<12}{'score':>10}{'blunders':>11}{'per game':>11}{'forfeits':>11}{'ms/move':>10}"
    print(header)
    print("-" * len(header))

    for mode in modes:
        t0 = time.time()
        total_score = total_bl = total_ff = total_plies = 0
        for g in range(args.games):
            s, bl, ff, pl = play_game(engine, net, device, mode, seed=1000 + g,
                                      max_plies=args.max_plies, depth=args.depth)
            total_score += s
            total_bl += bl
            total_ff += ff
            total_plies += pl
        ms = (time.time() - t0) / max(total_plies, 1) * 1000
        print(f"{mode:<12}{total_score/args.games:>10.3f}{total_bl:>11}"
              f"{total_bl/args.games:>11.2f}{total_ff:>11}{ms:>10.0f}", flush=True)

    engine.quit()


if __name__ == "__main__":
    main()