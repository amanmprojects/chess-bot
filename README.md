# chess-bot

A policy+value transformer that learns chess from human games (AlphaZero-style
representation, supervised learning) and plays vs Stockfish.

## Pipeline

    1. prepare.py          PGN.zst -> fixed-width records (features + policy slot + value)
    2. precompute_masks.py records  -> legal-move slot lists (once, kills the CPU bottleneck)
    3. train.py            records + masks -> policy+value transformer checkpoint
    4. play.py             checkpoint -> holdout accuracy, games vs Stockfish

## Data record (69 bytes, see features.py)

    pieces   uint8[64]  piece id per square (0 empty, 1-6 white PNBRQK, 7-12 black)
    aux      uint8[2]   side/castling bits, en-passant file
    policy   uint16     move slot in the 64x73 space (0..4671)
    value    int8       result from the side-to-move's view (+1/0/-1)

Move slots: 73 labels per square -- 56 queen slides (8 dirs x 7 dist), 8 knight
jumps, 9 underpromotions (3 destinations x N/B/R). Castling = king slide 2;
en passant = pawn diagonal 1; queen promotion = slide 1. Illegal slots are
masked to -inf at train and play time.

## Model

    ChessNet: 64 square-token embeddings + 3 aux tokens (side/castling/ep),
    67 learned position vectors, 7 prenorm transformer blocks (d=256, 8 heads),
    policy head: shared Linear(d -> 73) per square -> 4672 logits,
    value head: mean-pool squares -> Linear -> tanh.
    5.58M params at this default. Trains in a few hours on an RTX 4060
    (AMP, AdamW, cosine LR). `d=320, n_layers=8` gives 9.92M -- more accurate
    per epoch but ~40% slower, so fewer epochs fit in the same wall clock.

## Usage

    # 1. extract (streams the .zst, ~2-3h for a full month)
    python prepare.py --input ~/code/llm/data/lichess_db_standard_rated_2023-01.pgn.zst \
                      --out-dir data

    # 2. precompute legal-move masks (once, ~10 min for 3M records)
    python precompute_masks.py --data-dir data

    # 3. train
    python train.py --data-dir data --epochs 10 --batch-size 1024 --lr 4e-4 --out data/ckpt.pt

    # 4. evaluate
    python play.py --ckpt data/ckpt.pt --mode eval
    python play.py --ckpt data/ckpt.pt --mode stockfish --games 20 --depth 8

    # quick pipeline check on a small corpus
    python prepare.py --input ~/code/llm/data/dev.pgn.zst --out-dir data_dev
    python train.py --data-dir data_dev --epochs 1 --eval-every 60 --d 128 --n-layers 4

## Play it in the browser

The `chess/` submodule is a zero-dependency static chess game whose built-in AI
is alpha-beta search in a Web Worker. The net cannot run there -- it needs CUDA
and a 67MB checkpoint -- so it runs in a Python process and the game calls it
over HTTP.

    git submodule update --init          # once
    python serve_model.py                # terminal 1, port 8001
    cd chess && node server.mjs          # terminal 2, serves the page

Then pick **Neural net (server)** in the level dropdown. Every move logs which
engine produced it; the net reports 1 node because it looks at exactly one
position, while any alpha-beta level reports thousands.

## Filters (prepare.py)

Both players >= 1800 Elo, base time >= 180s (no bullet), Termination == Normal,
moves in [12, 250]. Games replayed with python-chess; corruption > 0.5% aborts.
