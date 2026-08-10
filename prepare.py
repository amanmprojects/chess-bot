"""Stream a lichess PGN.zst -> fixed-width training records (69 bytes each).

One record per sampled mid-game position of a qualifying game: board features +
the human's move slot + the game result from the side-to-move's view.

The bulk pass is a hand-rolled line filter (regex strip + header parse) so that
python-chess is only invoked on games that survive the filters -- this keeps a
full-month pass at ~2-3h instead of 8h+ (same trick as the chessLLM prep).
Legality is guaranteed by construction: every surviving game is replayed with
board.push_san, and corruption is counted and fatal if it exceeds 0.5%.

Filters (hard by default, matching the chessLLM prep):
    both Elo >= --min-elo, base time >= --min-time-control (no bullet),
    Termination == Normal, result != '*', moves in [--min-moves, --max-moves]

Usage:
    python prepare.py --input ~/code/llm/data/dev.pgn.zst --out-dir data --max-records 50000
    python prepare.py --input ~/code/llm/data/lichess_db_standard_rated_2023-01.pgn.zst \
                      --out-dir data
"""

import argparse
import io
import random
import re
import time
from pathlib import Path

import chess
import numpy as np
import zstandard

from features import DTYPE, board_to_features, move_to_slot

STRIP = re.compile(r"\{[^}]*\}|\$\d+|\d+\.{1,3}|[?!]+")
RESULT = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")
HEADER = re.compile(r"^\[([A-Za-z]+)\s+\"([^\"]*)\"\s*\]\s*$")


def iter_games(path):
    """Yield (headers, movetext) for every game, streaming from the .zst."""
    with zstandard.ZstdDecompressor().stream_reader(open(path, "rb")) as reader:
        txt = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        hdr, mt = {}, []
        for line in txt:
            if line.startswith("["):
                m = HEADER.match(line.strip())
                if m:
                    hdr[m.group(1)] = m.group(2)
                mt = []
                continue
            if not line.strip():
                if mt:
                    yield hdr, "".join(mt)
                    hdr, mt = {}, []
                continue
            if hdr or mt:
                mt.append(line)
        if mt:
            yield hdr, "".join(mt)


def result_value(result):
    return {"1-0": 1, "0-1": -1}.get(result, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--min-elo", type=int, default=1800)
    ap.add_argument("--min-time-control", type=int, default=180)
    ap.add_argument("--min-moves", type=int, default=12)
    ap.add_argument("--max-moves", type=int, default=250)
    ap.add_argument("--max-records", type=int, default=3_000_000)
    ap.add_argument("--records-per-game", type=int, default=2)
    ap.add_argument("--min-ply", type=int, default=8,
                    help="earliest ply to sample from (0 = include openings)")
    ap.add_argument("--max-ply", type=int, default=0,
                    help="latest ply to sample from (0 = no cap)")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=200_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_fh = open(out_dir / "train.bin", "wb")
    val_fh = open(out_dir / "val.bin", "wb")

    games = corrupt = total = 0
    t0 = time.time()
    for hdr, mt in iter_games(args.input):
        games += 1
        text = RESULT.sub("", mt)
        text = STRIP.sub("", text)
        moves = " ".join(text.split()).split()
        if not (args.min_moves <= len(moves) <= args.max_moves):
            continue
        try:
            if int(hdr.get("WhiteElo", 0)) < args.min_elo or int(hdr.get("BlackElo", 0)) < args.min_elo:
                continue
        except (TypeError, ValueError):
            continue
        tc = hdr.get("TimeControl", "")
        base = tc.split("+")[0]
        if base.endswith("s") or base == "-":
            continue
        try:
            if base and int(base) < args.min_time_control:
                continue
        except ValueError:
            continue
        if hdr.get("Termination", "Normal") != "Normal":
            continue

        board = chess.Board()
        hist = [board.copy()]
        move_objs = []
        try:
            for san in moves:
                mv = board.push_san(san)
                move_objs.append(mv)
                hist.append(board.copy())
        except ValueError:
            corrupt += 1
            continue
        if len(hist) < args.min_ply + 2:
            continue

        result = result_value(hdr.get("Result", ""))
        hi = len(hist) - 1 if not args.max_ply else min(args.max_ply, len(hist) - 1)
        if hi <= args.min_ply:
            continue
        picks = rng.sample(range(args.min_ply, hi),
                           min(args.records_per_game, hi - args.min_ply))
        for i in picks:
            pos, target = hist[i], move_objs[i]
            pieces, aux = board_to_features(pos)
            slot = move_to_slot(pos, target)
            side_sign = 1 if pos.turn == chess.WHITE else -1
            rec = np.array((pieces, aux, slot, result * side_sign), dtype=DTYPE)
            (val_fh if total % args.val_every == 0 else train_fh).write(rec.tobytes())
            total += 1
            if total >= args.max_records:
                break
        if total >= args.max_records:
            break
        if games % args.log_every == 0:
            rate = total / (time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] games {games:,} records {total:,} "
                  f"corrupt {corrupt:,} rate {rate:.0f}/s", flush=True)

    train_fh.close()
    val_fh.close()
    dt = time.time() - t0
    rate = corrupt / games * 100 if games else 0
    if rate > 0.5:
        raise SystemExit(f"[fatal] {corrupt:,}/{games:,} games corrupt ({rate:.2f}%)")
    train_size = (out_dir / "train.bin").stat().st_size
    val_size = (out_dir / "val.bin").stat().st_size
    print(f"[done] {games:,} games in {dt/60:.1f} min  records {total:,}  "
          f"train {train_size/1e6:.1f} MB  val {val_size/1e6:.1f} MB  "
          f"corrupt {rate:.2f}%")


if __name__ == "__main__":
    main()
