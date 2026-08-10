"""Rough Elo estimate: play the model against calibrated Stockfish opponents.

Anchors on Stockfish's UCI_Elo (calibrated, floor 1320) and, below that floor,
on Skill Level 0-2. Converts the match score to a rating via the logistic curve:

    elo_model = elo_opp + 400 * log10(S / (1 - S))

Two deliberate choices:

* **Random opening plies.** pick_move at temperature 0 is deterministic, so
  every game from the start position would be identical. Each game starts from
  a short random legal opening so the sample is not one game repeated N times.
* **Move cap + adjudication.** A weak model can shuffle for hundreds of plies.
  Games past --max-plies are adjudicated by Stockfish's own evaluation, which
  is what a human would do rather than scoring a dead-lost position as a draw.

    python estimate_elo.py --games 20 --opponents 1320
"""

import argparse
import math
import random

import chess
import chess.engine
import torch

from model import ChessNet
from play import pick_move


def wilson(s, n, z=1.96):
    """Score-rate confidence interval. Normal approx is wrong near 0 and 1,
    which is exactly where a weak engine lands."""
    if n == 0:
        return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return max(c - m, 0.0), min(c + m, 1.0)


def score_to_elo(score_rate, opp_elo):
    if score_rate <= 0.0:
        return None          # below measurable range
    if score_rate >= 1.0:
        return None          # above measurable range
    return opp_elo + 400 * math.log10(score_rate / (1 - score_rate))


def random_opening(rng, plies):
    """A short random legal opening so deterministic play still yields variety."""
    b = chess.Board()
    for _ in range(plies):
        if b.is_game_over():
            break
        b.push(rng.choice(list(b.legal_moves)))
    return b


def play_match(net, engine, device, games, opening_plies, max_plies,
               movetime, rng, label):
    score = 0.0
    wins = draws = losses = 0
    for g in range(games):
        board = random_opening(rng, opening_plies)
        if board.is_game_over():
            continue
        model_white = (g % 2 == 0)
        plies = 0
        while not board.is_game_over() and plies < max_plies:
            if (board.turn == chess.WHITE) == model_white:
                mv = pick_move(net, board, device, 0.0)
                if isinstance(mv, tuple):
                    mv = mv[0]
                if mv not in board.legal_moves:      # must never happen
                    raise RuntimeError(f"illegal move {mv} in {board.fen()}")
            else:
                mv = engine.play(board, chess.engine.Limit(time=movetime)).move
            board.push(mv)
            plies += 1

        if board.is_game_over():
            res = board.result()
        else:
            # Adjudicate by Stockfish's eval from the model's point of view.
            info = engine.analyse(board, chess.engine.Limit(time=movetime))
            cp = info["score"].white().score(mate_score=10000)
            res = "1-0" if cp > 150 else "0-1" if cp < -150 else "1/2-1/2"

        if res == "1/2-1/2":
            score += 0.5
            draws += 1
        elif (res == "1-0") == model_white:
            score += 1.0
            wins += 1
        else:
            losses += 1
        print(f"  [{label}] game {g+1}/{games} "
              f"{'W' if model_white else 'B'} {res} "
              f"(running {score:.1f}/{g+1})", flush=True)
    return score, wins, draws, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpt.pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opponents", default="1320",
                    help="comma list; ints = UCI_Elo, 'skillN' = Skill Level N")
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = ChessNet(d=256, n_layers=7, n_heads=8).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model {args.ckpt} (step {ck.get('step')})  device {device}")
    print(f"{args.games} games/opponent, {args.movetime*1000:.0f}ms/move, "
          f"{args.opening_plies} random opening plies, cap {args.max_plies}\n")

    rows = []
    for spec in args.opponents.split(","):
        spec = spec.strip()
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        try:
            if spec.startswith("skill"):
                lvl = int(spec[5:])
                engine.configure({"Skill Level": lvl})
                # Skill Level is not a calibrated rating. These are rough
                # community-reported equivalents, used only to bracket a model
                # that scores 0 against the UCI_Elo floor of 1320.
                opp_elo = {0: 700, 1: 800, 2: 900, 3: 1000,
                           4: 1100, 5: 1200}.get(lvl, 800)
                approx = True
            else:
                opp_elo = int(spec)
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": opp_elo})
                approx = False
            rng = random.Random(args.seed)
            print(f"vs {spec} (~{opp_elo} Elo{'*' if approx else ''}):")
            s, w, d, l = play_match(net, engine, device, args.games,
                                    args.opening_plies, args.max_plies,
                                    args.movetime, rng, spec)
        finally:
            engine.quit()

        n = w + d + l
        rate = s / max(n, 1)
        lo, hi = wilson(s, n)
        rows.append((spec, opp_elo, approx, s, n, w, d, l, rate,
                     score_to_elo(rate, opp_elo),
                     score_to_elo(lo, opp_elo), score_to_elo(hi, opp_elo)))
        print(f"  => {s:.1f}/{n}  (+{w} ={d} -{l})  {rate*100:.0f}%\n")

    print(f"{'opponent':<10}{'opp_elo':>9}{'score':>10}{'rate':>8}"
          f"{'est_elo':>10}{'95% CI':>18}")
    for (spec, oe, approx, s, n, w, d, l, rate, est, lo, hi) in rows:
        est_s = f"{est:.0f}" if est is not None else ("<range" if rate <= 0
                                                      else ">range")
        ci = (f"{lo:.0f}-{hi:.0f}" if lo is not None and hi is not None
              else (f"<{hi:.0f}" if hi is not None else "wide"))
        print(f"{spec:<10}{oe:>9}{f'{s:.1f}/{n}':>10}{rate*100:>7.0f}%"
              f"{est_s:>10}{ci:>18}"
              f"{'  (*approx anchor)' if approx else ''}")

    print("\nCaveats: no search (one forward pass per move), fast time control,"
          "\nsmall sample, and Skill-Level anchors are approximate. Treat this"
          "\nas a band, not a rating.")


if __name__ == "__main__":
    main()
