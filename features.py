"""Frozen feature and move-slot encoding shared by prepare.py, train.py and play.py.

Record layout (69 bytes, fixed width, memmap-friendly numpy dtype):

    pieces   uint8[64]   one id per square, row-major from a8 to h1
    aux      uint8[2]    [0] bit0=side(1=white), bit1..4=castling WK,WQ,BK,BQ
                         [1] en-passant file 0..7, or 15 when none
    policy   uint16      the move's slot in the 64x73 space (0..4671)
    value    int8        game result from the side-to-move's view: +1 win, 0 draw, -1 loss

Piece ids: 0 empty, 1-6 white P N B R Q K, 7-12 black.

Move slots: 73 labels per square, applied to the move's FROM square:
    0..55    queen-style slides: 8 directions x 7 distances
             (covers pawn pushes/captures, double pushes, king moves, castling,
              en passant captures, queen promotions)
    56..63   knight jumps (8 fixed offsets)
    64..72   underpromotions: 3 destination files x 3 pieces (N=0, B=1, R=2)
"""

import numpy as np
import chess

DTYPE = np.dtype([("pieces", "u1", (64,)), ("aux", "u1", (2,)),
                  ("policy", "u2"), ("value", "i1")])
RECORD_BYTES = DTYPE.itemsize
POLICY_SIZE = 64 * 73

PIECE_ID = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
PIECE_CHAR = {v: k for k, v in PIECE_ID.items()}
for c in "PNBRQK":
    PIECE_ID[c.lower()] = PIECE_ID[c] + 6
PIECE_CHAR.update({v + 6: c.lower() for v, c in enumerate("PNBRQK", 1)})

DIR_OFFSET = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1),
              "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}
DIR_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
KNIGHT_OFFSETS = ((1, 2), (2, 1), (2, -1), (1, -2),
                  (-1, -2), (-2, -1), (-2, 1), (-1, 2))
PROMO_PIECE_ID = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
EP_NONE = 15


def move_to_slot(board, move):
    """Map a legal move to its slot 0..4671 in the 64x73 policy space."""
    f0, r0 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f1, r1 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    df, dr = f1 - f0, r1 - r0
    if move.promotion and move.promotion != chess.QUEEN:
        dest_idx = df + 1
        label = 64 + dest_idx * 3 + PROMO_PIECE_ID[move.promotion]
    elif (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        label = 56 + KNIGHT_OFFSETS.index((df, dr))
    elif df == 0 and dr == 0:
        raise ValueError(f"null move {move} in {board.fen()}")
    else:
        norm = (df and df // abs(df), dr and dr // abs(dr))
        label = DIR_ORDER.index({v: k for k, v in DIR_OFFSET.items()}[norm]) * 7 \
            + max(abs(df), abs(dr)) - 1
    return move.from_square * 73 + label


def policy_mask(board):
    """Boolean mask over the 4672 slots, True for every legal move."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_slot(board, move)] = True
    return mask


def board_to_features(board):
    """Encode a board into the pieces (64) and aux (2) arrays."""
    pieces = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            pieces[sq] = PIECE_ID[piece.symbol()]
    aux0 = (1 if board.turn == chess.WHITE else 0)
    for right, bit in ((board.has_kingside_castling_rights(chess.WHITE), 1),
                       (board.has_queenside_castling_rights(chess.WHITE), 2),
                       (board.has_kingside_castling_rights(chess.BLACK), 3),
                       (board.has_queenside_castling_rights(chess.BLACK), 4)):
        if right:
            aux0 |= 1 << bit
    ep = board.ep_square
    aux1 = EP_NONE if ep is None else chess.square_file(ep)
    return pieces, np.array([aux0, aux1], dtype=np.uint8)


def features_to_board(pieces, aux):
    """Rebuild a python-chess board from the features (for masking at train time)."""
    rows = []
    for rank in range(7, -1, -1):
        cells, run = [], 0
        for file in range(8):
            pid = int(pieces[rank * 8 + file])
            if pid:
                if run:
                    cells.append(str(run))
                    run = 0
                cells.append(PIECE_CHAR[pid])
            else:
                run += 1
        if run:
            cells.append(str(run))
        rows.append("".join(cells))
    piece_placement = "/".join(rows)
    turn = "w" if aux[0] & 1 else "b"
    castling = "".join(k for k, bit in (("K", 1), ("Q", 2), ("k", 3), ("q", 4))
                       if aux[0] & (1 << bit)) or "-"
    ep_file = int(aux[1])
    ep = "-" if ep_file == EP_NONE else \
        chr(ord("a") + ep_file) + ("6" if turn == "w" else "3")
    return chess.Board(f"{piece_placement} {turn} {castling} {ep} 0 1")
