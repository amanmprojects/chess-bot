"""Stockfish-evaluate the existing training records, on Modal CPU.

The training records already store enough to rebuild the board: `pieces uint8[64]`
plus `aux uint8[2]` (side, castling, ep file). features.features_to_board() is a
clean inverse, so every one of the 2.6M positions can be re-evaluated with a real
engine without re-downloading any PGNs and without re-running prepare.py.

Why this matters: the shipped value head is trained on the game result alone
(+1/0/-1 from prepare.py), a label that says nothing about the position it was
played from. Replacing it with a Stockfish score is the precondition for search
over the net being worth anything at all.

Writes a parallel .evl file of int16 centipawns from the side-to-move's view,
clamped to +-EVAL_CLAMP so mate scores do not dominate the regression.
"""

import modal

app = modal.App("chess-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .uv_pip_install("numpy", "chess==1.11.2")
    .add_local_file("features.py", "/root/features.py")
)

vol = modal.Volume.from_name("chess-bot-data", create_if_missing=True)
VOL = "/vol"

# Centipawns are clamped here so that a mate in 1 (+30000 cp) is not 30x the
# magnitude of a routine +100cp advantage and dominate the MSE.
EVAL_CLAMP = 2000
NODE_LIMIT = 2000

RECORD_BYTES = 69
FIELDS = [("pieces", "u1", (64,)), ("aux", "u1", (2,)), ("policy", "u2"), ("value", "i1")]

# Debian's stockfish package installs the binary into /usr/games, which is not on
# PATH for a non-login shell inside the container.
STOCKFISH = "/usr/games/stockfish"


def eval_shard(shard: int, num_shards: int, src: str, dst: str, n_records: int,
               node_limit: int = NODE_LIMIT):
    """Evaluate every record where index % num_shards == shard.

    Sharding by stride rather than by contiguous block keeps each worker's
    engine warm against a representative mix of opening/middlegame/endgame
    positions instead of a homogeneous tail.
    """
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np
    import chess, chess.engine
    from features import features_to_board

    dtype = np.dtype(FIELDS)
    mm = np.memmap(f"{VOL}/{src}", dtype=dtype, mode="r", shape=(n_records,))

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    out = np.zeros(n_records, dtype=np.int16)

    t0 = time.time()
    done = 0
    for i in range(shard, n_records, num_shards):
        rec = mm[i]
        board = features_to_board(rec["pieces"], rec["aux"])
        try:
            info = engine.analyse(board, chess.engine.Limit(nodes=node_limit))
            score = info["score"].pov(board.turn).score(mate_score=EVAL_CLAMP * 10)
        except Exception:
            score = 0
        out[i] = max(-EVAL_CLAMP, min(EVAL_CLAMP, int(score)))
        done += 1
        if done % 2000 == 0:
            el = time.time() - t0
            print(f"[shard {shard}] {done} evals  {done/el:.1f}/s  "
                  f"eta {(n_records/num_shards - done)/(done/el)/60:.1f}m", flush=True)

    engine.quit()

    # Each shard owns a strided subset; write it back as its own file rather than
    # racing 64 workers on one memmap. Under evl/ so the whole set comes down
    # with one recursive `modal volume get` instead of 64 CLI round trips.
    import os
    os.makedirs(f"{VOL}/evl", exist_ok=True)
    sub = out[shard::num_shards]
    sub.tofile(f"{VOL}/evl/{dst}.{shard:03d}")
    vol.commit()
    return {"shard": shard, "n": int(len(sub)), "secs": round(time.time() - t0, 1)}


@app.function(image=image, cpu=8.0, memory=8192, timeout=6 * 3600,
              volumes={VOL: vol}, retries=modal.Retries(max_retries=3))
def worker(shard: int, num_shards: int, src: str, dst: str, n_records: int,
           node_limit: int = NODE_LIMIT):
    return eval_shard(shard, num_shards, src, dst, n_records, node_limit)


@app.local_entrypoint()
def main(src: str = "train.bin", dst: str = "train.evl", shards: int = 64,
         n_records: int = 2_587_000, node_limit: int = NODE_LIMIT):
    jobs = [worker.spawn(i, shards, src, dst, n_records, node_limit)
            for i in range(shards)]
    for j in jobs:
        r = j.get()
        print(f"done shard {r['shard']}: {r['n']} evals in {r['secs']}s", flush=True)