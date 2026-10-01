"""Download additional Lichess months and prepare training records, on Modal CPU.

The January 2023 dump alone was 9.1M games -> 2.6M records, and RUN_REPORT.md
found the model overfitting that from epoch ~10. Scaling the model without more
unique positions just reproduces the same ceiling earlier, so this exists to
multiply the record count.

Why it shards: prepare.py is single-threaded and took 100 minutes for one month
on a laptop. A .zst stream cannot be seeked, so this decompresses once, splits the
plain PGN into byte ranges on game boundaries, and runs the filter+encode step
across many workers in parallel. Same filters as prepare.py (both players >=
--min-elo, no bullet, Termination Normal, move-count bounds) and the same 69-byte
record layout, so the output drops straight into the existing training pipeline.

Cost shape: the download is the expensive part at $0.04/GiB egress (~1 month is
~30GiB compressed); CPU is cheap, and Volume storage is free for the first 1TiB.
"""

import re

import modal

app = modal.App("chess-prepare")

# Module-level, not inside prep_shard: encode_game uses them, and a name that
# only exists in the caller's frame is a NameError the moment a shard runs.
STRIP = re.compile(rb"\{[^}]*\}|\$\d+|\d+\.{1,3}|[?!]+")
RESULT = re.compile(rb"\s*(1-0|0-1|1/2-1/2|\*)\s*$")
HEADER = re.compile(rb'^\[([A-Za-z]+)\s+"([^"]*)"\]\s*$')
CLK = re.compile(rb"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
PIECE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
for _c in "PNBRQK":
    PIECE[_c.lower()] = PIECE[_c] + 6

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .uv_pip_install("numpy", "chess==1.11.2", "zstandard")
    .add_local_file("features.py", "/root/features.py")
)

vol = modal.Volume.from_name("chess-bot-data", create_if_missing=True)
VOL = "/vol"


@app.function(image=image, timeout=6 * 3600, memory=16384, volumes={VOL: vol})
def download(month: str, url: str, max_bytes: int = 0):
    """Stream a month's .pgn.zst to the Volume, decompressing on the way in.

    Storing the decompressed PGN rather than the .zst is deliberate: the shard
    workers need to seek, and zstd streams cannot be seeked.

    max_bytes stops early once that many *decompressed* bytes are on the volume.
    The filters keep ~10.4% of games at ~0.21 records/game, so 150 GB of prefix
    already covers 65M games = ~13.6M candidate records; the remaining 70 GB of
    the month would only add records this run has no time to train on, and the
    measured Lichess rate decays from ~38 to ~15 MB/s over the first 20 minutes,
    so the tail is the expensive part.
    """
    import os, zstandard, urllib.request

    out = f"{VOL}/pgn/{month}.pgn"
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f"[dl] {month} already present ({os.path.getsize(out)/1e9:.1f} GB)")
        return {"month": month, "bytes": os.path.getsize(out), "cached": True}

    os.makedirs(f"{VOL}/pgn", exist_ok=True)
    tmp = out + ".part"
    total = 0
    capped = False
    dctx = zstandard.ZstdDecompressor()
    with urllib.request.urlopen(url) as resp:
        with open(tmp, "wb") as fh:
            with dctx.stream_reader(resp) as reader:
                while True:
                    chunk = reader.read(8 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
                    if total % (2 << 30) < (8 << 20):
                        print(f"[dl] {month} {total/1e9:.1f} GB", flush=True)
                    if max_bytes and total >= max_bytes:
                        capped = True
                        break
    os.replace(tmp, out)
    vol.commit()
    tag = " (capped)" if capped else ""
    print(f"[dl] {month} done: {total/1e9:.1f} GB decompressed{tag}", flush=True)
    return {"month": month, "bytes": total, "cached": False, "capped": capped}


@app.function(image=image, timeout=6 * 3600, memory=16384, volumes={VOL: vol})
def index(month: str):
    """Record the byte offset of each game's opening '[', for sharding.

    The offsets go to a binary int64 file rather than JSON. A month is ~96M
    games; as a JSON array that is ~1GB which every one of 32 prep workers would
    json.load into ~3GB of Python ints (96GB total). A memmapped int64 array
    costs the worker nothing and loads in O(1).

    Matching b"\\n[Event " with bytes.find instead of iterating lines runs at
    C speed -- the line-by-line version needed ~45 min per month, which was the
    whole point of sharding in the first place.
    """
    import os, struct

    path = f"{VOL}/pgn/{month}.pgn"
    index_path = f"{VOL}/pgn/{month}.idx"
    if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
        print(f"[idx] {month} cached")
        return {"month": month, "games": os.path.getsize(index_path) // 8,
                "cached": True}

    marker = b"\n[Event "
    chunk = 1 << 26
    games = 0
    with open(path, "rb") as fh, open(index_path, "wb") as ofh:
        if fh.read(7) == b"[Event ":
            ofh.write(struct.pack("<q", 0))
            games += 1
        fh.seek(0)
        tail, pos = b"", 0
        pack = struct.Struct("<q").pack
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            data = tail + buf
            base = pos - len(tail)
            start = 0
            while True:
                i = data.find(marker, start)
                if i < 0:
                    break
                ofh.write(pack(base + i + 1))
                games += 1
                start = i + 1
            tail = data[-(len(marker) - 1):]
            pos += len(buf)
    vol.commit()
    print(f"[idx] {month}: {games:,} games, {pos/1e9:.1f} GB", flush=True)
    return {"month": month, "games": games, "cached": False}


def iter_games_in(path: str, start: int, end: int):
    """Yield (headers, movetext_bytes) for every game in byte range [start, end).

    Mirrors prepare.py:iter_games, but bounded and byte-oriented, because a shard
    only owns a slice of the file. Module-level rather than inline in prep_shard
    so the parser can be exercised against real PGN without a Modal round trip.
    """
    buf_hdr, buf_moves = {}, []
    with open(path, "rb") as fh:
        fh.seek(start)
        pos = start
        while pos < end:
            line = fh.readline()
            if not line:
                break
            pos += len(line)
            if line.startswith(b"["):
                m = HEADER.match(line.strip())
                if m:
                    buf_hdr[m.group(1)] = m.group(2)
                buf_moves = []
            elif not line.strip():
                # Only a blank line that *terminates movetext* ends the game.
                # PGN also puts one between the headers and the moves (line 17
                # of every real Lichess game); resetting there wipes buf_hdr
                # before encode_game sees it and drops every move line, so the
                # shard would report 0 games kept and silently produce nothing.
                if buf_moves:
                    yield buf_hdr, b"".join(buf_moves)
                    buf_hdr, buf_moves = {}, []
            elif buf_hdr or buf_moves:
                buf_moves.append(line)
        if buf_moves:
            yield buf_hdr, b"".join(buf_moves)


@app.function(image=image, cpu=4.0, memory=8192, timeout=6 * 3600,
              volumes={VOL: vol}, retries=modal.Retries(max_retries=2))
def prep_shard(month: str, shard: int, num_shards: int, min_elo: int,
               max_records_per_game: int, min_ply: int, min_tc: int,
               min_clk: float, clk_frac: float, val_every: int = 128,
               keep_frac: float = 1.0):
    """Filter + encode one shard of the month's games into 69-byte records.

    Train/val is split by *game*, not by record: prepare.py's records come from
    `random.sample` over one game, so a record-level split would put two views of
    the same position in both halves and report an optimistic val loss. Taking
    every val_every-th game keeps the two halves disjoint and identically
    distributed.

    keep_frac samples the kept pool down to a target size. Subsampling after the
    fact would need the same rules reapplied to a 69-byte record stream with no
    game boundaries left in it, so it is decided here while the game is still
    whole.
    """
    import sys, os, time, random
    sys.path.insert(0, "/root")
    import numpy as np
    import chess
    from features import board_to_features, move_to_slot

    random.seed(f"{month}:{shard}")

    offs = np.memmap(f"{VOL}/pgn/{month}.idx", dtype="<i8", mode="r")
    n = len(offs)
    size = os.path.getsize(f"{VOL}/pgn/{month}.pgn")
    lo = (n * shard) // num_shards
    hi = (n * (shard + 1)) // num_shards
    start = int(offs[lo]) if lo < n else size
    end = int(offs[hi]) if hi < n else size

    DT = np.dtype([("pieces", "u1", (64,)), ("aux", "u1", (2,)),
                   ("policy", "u2"), ("value", "i1")])

    import os
    os.makedirs(f"{VOL}/recs", exist_ok=True)
    train_path = f"{VOL}/recs/{month}.{shard:03d}.train.bin"
    val_path = f"{VOL}/recs/{month}.{shard:03d}.val.bin"
    games = kept = corrupt = total = val_total = 0
    t0 = time.time()

    with open(train_path, "wb") as out_tr, open(val_path, "wb") as out_va:
        for hdr, movetext in iter_games_in(f"{VOL}/pgn/{month}.pgn", start, end):
            games += 1
            if keep_frac < 1.0 and random.random() >= keep_frac:
                continue
            dest = out_va if games % val_every == 0 else out_tr
            ok, nrec = encode_game(hdr, movetext, dest, DT, min_elo,
                                   max_records_per_game, min_ply, PIECE,
                                   min_tc, min_clk, clk_frac)
            if ok == "ok":
                kept += 1
                total += nrec
                if dest is out_va:
                    val_total += nrec
            elif ok == "corrupt":
                corrupt += 1

    vol.commit()
    print(f"[shard {shard}] games {games:,} kept {kept:,} corrupt {corrupt:,} "
          f"train {total - val_total:,} val {val_total:,} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)
    return {"shard": shard, "games": games, "records": total,
            "val_records": val_total, "corrupt": corrupt}


def encode_game(hdr, movetext, out, DT, min_elo, per_game, min_ply, PIECE,
                min_tc=180, min_clk=10.0, clk_frac=0.05):
    """Apply prepare.py's filters plus the time-trouble filter. Mirrors that file.

    The extra filter is the clock one. prepare.py already drops every game whose
    Termination != "Normal" — measured on a 998,720-game Lichess sample, that is
    35% of all 1800+ games (104,780 "Time forfeit" of 299,140), so outright flag
    losses are already gone. What is left is the game where a player reached 3
    seconds, survived, and resigned a won position. Lichess writes `[%clk]` after
    every move (present on 99.9% of games), so the per-move clock *is* available;
    the base-time floor was only ever a proxy for it.

    Raising that floor instead is the wrong lever: base >= 600s keeps 19.0% of
    the pool (24,427 of 128,600), while min_clk >= max(10s, 5% of base) keeps
    85.7%. The cheap base floor threw away the data this run exists to add.
    """
    import random
    import numpy as np
    import chess
    from features import board_to_features, move_to_slot

    text = RESULT.sub(b"", movetext)
    text = STRIP.sub(b"", text)
    moves = " ".join(text.decode("utf-8", "ignore").split()).split()
    if not (12 <= len(moves) <= 250):
        return "skip", 0
    try:
        if int(hdr.get(b"WhiteElo", b"0")) < min_elo or \
           int(hdr.get(b"BlackElo", b"0")) < min_elo:
            return "skip", 0
    except (TypeError, ValueError):
        return "skip", 0
    tc = hdr.get(b"TimeControl", b"").decode("utf-8", "ignore")
    base = tc.split("+")[0]
    if base.endswith("s") or base == "-":
        return "skip", 0
    try:
        if base and int(base) < min_tc:
            return "skip", 0
        base_s = float(base)
    except ValueError:
        return "skip", 0
    if hdr.get(b"Termination", b"Normal") != b"Normal":
        return "skip", 0

    # Per-move clock, read before STRIP deletes the { ... } that carries it.
    clocks = [int(h) * 3600 + int(m) * 60 + float(s)
              for h, m, s in CLK.findall(movetext)]
    if not clocks:
        return "skip", 0
    if min(clocks) < max(min_clk, clk_frac * base_s):
        return "skip", 0

    board = chess.Board()
    hist = [board.copy()]
    move_objs = []
    try:
        for san in moves:
            mv = board.push_san(san)
            move_objs.append(mv)
            hist.append(board.copy())
    except ValueError:
        return "corrupt", 0
    if len(hist) < min_ply + 2:
        return "skip", 0

    result = {"1-0": 1, "0-1": -1}.get(hdr.get(b"Result", b"").decode(), 0)
    hi = len(hist) - 1
    if hi <= min_ply:
        return "skip", 0
    picks = random.sample(range(min_ply, hi), min(per_game, hi - min_ply))
    n = 0
    for i in picks:
        pos, target = hist[i], move_objs[i]
        pieces, aux = board_to_features(pos)
        slot = move_to_slot(pos, target)
        side_sign = 1 if pos.turn == chess.WHITE else -1
        rec = np.array((pieces, aux, slot, result * side_sign), dtype=DT)
        out.write(rec.tobytes())
        n += 1
    return "ok", n


@app.function(image=image, volumes={VOL: vol}, timeout=30 * 60)
def cleanup(month: str):
    """Drop the decompressed PGN once its records exist.

    A month is ~230 GB decompressed; leaving three on the volume is both slow to
    index and the only expensive thing here. Storage is bounded to one month at
    a time as a result.
    """
    import os
    freed = 0
    for suffix in (".pgn", ".pgn.part", ".idx"):
        p = f"{VOL}/pgn/{month}{suffix}"
        if os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
    vol.commit()
    print(f"[clean] {month} freed {freed/1e9:.1f} GB")
    return {"month": month, "freed": freed}


@app.function(image=image, timeout=6 * 3600, memory=8192, volumes={VOL: vol})
def concat(split: str, dst: str):
    """Glue recs/*.{split}.bin into one contiguous file for eval and training.

    Shard files are sorted so the order is (month, shard), which keeps each
    month's games contiguous -- a strided subsample later then never picks two
    records from the same game.
    """
    import glob, os, shutil
    files = sorted(glob.glob(f"{VOL}/recs/*.{split}.bin"))
    if not files:
        raise FileNotFoundError(f"no {VOL}/recs/*.{split}.bin")
    if os.path.exists(dst):
        os.remove(dst)
    nbytes = 0
    with open(dst, "wb") as out:
        for f in files:
            with open(f, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 24)
            nbytes += os.path.getsize(f)
    vol.commit()
    if nbytes % 69:
        raise ValueError(f"{dst}: {nbytes} bytes is not a multiple of 69")
    print(f"[cat] {split} -> {dst} {nbytes//69:,} records "
          f"({len(files)} shards, {nbytes/1e6:.1f} MB)", flush=True)
    return {"split": split, "files": len(files), "records": nbytes // 69}


@app.function(image=image, timeout=30 * 60, volumes={VOL: vol})
def finalize(month: str):
    """Promote a partial .part to .pgn so a stopped download is still usable.

    The stream is sequential, so a prefix of the file is a valid PGN: every game
    except possibly the last is complete. Cutting the download off on a deadline
    and indexing what arrived costs one truncated game, where restarting it
    throws away every byte already fetched.
    """
    import os
    part, out = f"{VOL}/pgn/{month}.pgn.part", f"{VOL}/pgn/{month}.pgn"
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        return {"month": month, "bytes": os.path.getsize(out), "cached": True}
    if not os.path.exists(part):
        raise FileNotFoundError(f"neither {out} nor {part} exists")
    os.replace(part, out)
    vol.commit()
    size = os.path.getsize(out)
    print(f"[fin] {month}: {size/1e9:.1f} GB prefix promoted", flush=True)
    return {"month": month, "bytes": size, "cached": False}


@app.local_entrypoint()
def main(months: str = "2023-02", shards: int = 32, min_elo: int = 1800,
         per_game: int = 2, min_ply: int = 8, min_tc: int = 180,
         min_clk: float = 10.0, clk_frac: float = 0.05, val_every: int = 128,
         keep_frac: float = 1.0, max_bytes: int = 0, stage: str = "all"):
    """stage = download | index | prep | concat | clean | all

    min_tc stays at prepare.py's 180 so old and new records are drawn from the
    same game pool; the timeout exclusion is done by min_clk/clk_frac instead,
    which measures the clock directly rather than guessing it from the base time.
    """
    base = "https://database.lichess.org/standard/lichess_db_standard_rated_{m}.pgn.zst"
    ms = [m.strip() for m in months.split(",") if m.strip()]

    if stage in ("all", "download"):
        for r in download.map(ms, [base.format(m=m) for m in ms],
                              [max_bytes] * len(ms)):
            print("[dl]", r, flush=True)
    if stage in ("all", "index"):
        for r in index.map(ms):
            print("[idx]", r, flush=True)
    if stage in ("all", "prep"):
        # All months at once: prep is single-threaded per worker, so the wall
        # clock is set by the longest month / shards, not by three trips in
        # sequence.
        jobs = [(m, prep_shard.spawn(m, i, shards, min_elo, per_game, min_ply,
                                     min_tc, min_clk, clk_frac, val_every,
                                     keep_frac))
                for m in ms for i in range(shards)]
        train = val = gseen = 0
        for m, j in jobs:
            r = j.get()
            train += r["records"] - r["val_records"]
            val += r["val_records"]
            gseen += r["games"]
            print(f"[prep {m} #{r['shard']}] {r['records']:,}", flush=True)
        print(f"[prep] train {train:,} val {val:,} from {gseen:,} games "
              f"({train + val:,} total)", flush=True)
    if stage in ("all", "concat"):
        for split, dst in (("train", f"{VOL}/train2.bin"),
                           ("val", f"{VOL}/val2.bin")):
            print("[cat]", concat.remote(split, dst), flush=True)
    if stage == "finalize":
        for r in finalize.map(ms):
            print("[fin]", r, flush=True)
    if stage in ("all", "clean"):
        for r in cleanup.map(ms):
            print(r, flush=True)