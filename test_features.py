import chess
import numpy as np
from features import board_to_features, features_to_board, move_to_slot, policy_mask

fen = "r3k2r/1P4pp/2n5/2p1Pp2/8/8/5p1P/R3K2R w KQkq f6 0 1"
b = chess.Board(fen)
pieces, aux = board_to_features(b)
b2 = features_to_board(pieces, aux)
assert b2.fen() == b.fen(), (b2.fen(), b.fen())

slots = set()
for m in b.legal_moves:
    s = move_to_slot(b, m)
    assert 0 <= s < 4672
    assert s not in slots, f"duplicate slot {s} for {m}"
    slots.add(s)
mask = policy_mask(b)
assert all(mask[move_to_slot(b, m)] for m in b.legal_moves)
assert mask.sum() == len(list(b.legal_moves)) == len(slots)

b3 = chess.Board()
e4 = move_to_slot(b3, chess.Move.from_uci("e2e4"))
assert e4 == chess.parse_square("e2") * 73 + 1, e4
sf = move_to_slot(b3, chess.Move.from_uci("g1f3"))
assert sf == chess.parse_square("g1") * 73 + 63, sf
promo = move_to_slot(b3, chess.Move.from_uci("e7e8n"))
assert promo == chess.parse_square("e7") * 73 + 67, promo

b4 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
castle_k = move_to_slot(b4, next(m for m in b4.legal_moves if m.uci() == "e1g1"))
castle_q = move_to_slot(b4, next(m for m in b4.legal_moves if m.uci() == "e1c1"))
assert castle_k == chess.parse_square("e1") * 73 + 2 * 7 + 1
assert castle_q == chess.parse_square("e1") * 73 + 6 * 7 + 1

b5 = chess.Board("4k3/8/8/8/Pp6/8/8/4K3 b - b3 0 1")
ep = next(m for m in b5.legal_moves if m.to_square == chess.parse_square("b3"))
assert move_to_slot(b5, ep) == chess.parse_square("b4") * 73 + 4 * 7

for move in [chess.Move.from_uci("h2h3"), chess.Move.from_uci("b2b3")]:
    s = move_to_slot(b3, move)
    assert s == move.from_square * 73 + 0, (move, s)

print("all feature tests passed")
