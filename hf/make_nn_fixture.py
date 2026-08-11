"""Regenerate chess/test/fixtures/neural.json — the ground truth for the
browser-side port of features.py (see chess/src/neural.js).

The JS game re-implements board_to_features, move_to_slot and the value->cp
formula so the policy net can run in a static page. This script snapshots the
python-chess outputs for a spread of positions; the node test suite then
asserts the JS port reproduces them exactly.

    ~/code/llm/.venv/bin/python hf/make_nn_fixture.py
"""

import json
import math
import pathlib
import sys

import chess

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features import board_to_features, move_to_slot  # noqa: E402

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",   # en passant
    "8/2P5/8/8/8/8/8/k1K5 w - - 0 1",                                   # underpromotions
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",        # pinned queen sac
    "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1",                                 # white kingside only
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",                             # black to move, all rights
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/8/8/8/2k5/1p6/1K6 b - - 0 1",
]

OUT = ROOT / "chess" / "test" / "fixtures" / "neural.json"


def main():
    positions = []
    for fen in FENS:
        board = chess.Board(fen)
        pieces, aux = board_to_features(board)
        slots = {m.uci(): int(move_to_slot(board, m)) for m in board.legal_moves}
        assert len(slots) == len(set(slots.values())), f"slot collision in {fen}"
        positions.append({
            "fen": fen,
            "pieces": [int(x) for x in pieces],
            "aux": [int(x) for x in aux],
            "slots": slots,
        })

    # Value head -> centipawns, the exact formula from serve_model.py.
    cps = []
    for v in (0.999, 0.9, 0.5, 0.25, 0.0, -0.25, -0.5, -0.9, -0.999):
        vv = max(min(v, 0.999), -0.999)
        cps.append([v, int(-400 * math.log10(2 / (vv + 1) - 1))])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"positions": positions, "cps": cps}, indent=1))
    print(f"wrote {OUT}: {len(positions)} positions, {sum(len(p['slots']) for p in positions)} moves")


if __name__ == "__main__":
    main()
