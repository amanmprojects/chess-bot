# Overnight run — 2026-08-10

## Short version

The model trained successfully and is ready: **`data/ckpt.pt`**, full-split
top1 **43.1%**, holdout top1 **43.7%** with **100% legal moves**. From-scratch
data, one supervisor attempt, zero training crashes.

The move encoding was **not** the bug that killed the previous 3-hour run. The
real cause was a crash in the *shutdown* path plus a supervisor that mistook a
stale log line for success. Details in `BUGS_FIXED.md`.

Two new bugs (F, G) surfaced this morning in `play.py` and are fixed and
verified — see "What broke this morning" below.

## What was wrong before

| bug | where | effect |
|---|---|---|
| A | `train.py:177` | `idx` NameError in the final eval |
| B | `train.py:177-182` | weights saved *after* eval, so the crash discarded them |
| C | `supervise_train.sh:20` | grepped the whole log for `[done]`, so a past run's success masked a new failure |

Together: train 3h → final eval crashes → no `[done]` → resume → `step ==
total_steps` so the loop body never runs → crash again → 20 attempts → exit.
The model was fine; the exit path destroyed it.

**The encoding was audited and is correct** — exhaustive bijection test over
20K legal moves, all promotions, all 64 from-squares, zero collisions. The
reported from-square bug was already fixed in `features.py:60`.

## Data (regenerated from scratch)

Source `lichess_db_standard_rated_2023-01.pgn.zst`, all previous data cleared.

    9,118,448 games in 100.5 min -> 2,600,000 records, 0.00% corrupt
    train 2,587,000 / val 13,000

Validation passed: 64/64 from-squares used, 0 target-in-mask failures, 0
mask mismatches over 4,000 sampled positions.

## Result

Scored on the **full 13,000-position val split** (1σ on top1 = 0.43pp):

| checkpoint | step | epoch | CE | top1 |
|---|---|---|---|---|
| `ckpt.best-24000.pt` | 24,000 | ~9.5 | **1.9005** | 0.4247 |
| `ckpt.best-38000.pt` | 38,000 | ~15 | 1.9675 | 0.4312 |
| `ckpt.best-40000.pt` = `ckpt.pt` | 40,000 | ~16 | 1.9570 | **0.4313** |
| `ckpt.pt.final` | 55,572 | 22 | 2.1619 | 0.4152 |

**Ship `ckpt.pt`** — it already holds the step-40,000 weights, so there is
nothing to move. It wins top1, the metric a move-playing bot is judged on. If
calibration ever matters more, step 24,000 is 0.057 better on CE.

Independent holdout check on the shipped model (3,000 positions, separate
sampler): **top1 0.4370, legal 1.0000**.

Three things worth knowing:

- **Running to completion made the model worse.** The final checkpoint is the
  worst of the four — 1.6pp below step 40,000. `best_acc` gating is the only
  reason the good weights survived.
- **38,000 and 40,000 are a tie**, 0.02pp = 0.0σ. The training-time eval ranked
  them 0.4463 vs 0.4443 and made 40,000 look decisive; on the full split that
  ordering is noise.
- **Quote ~43%, not ~44.5%.** The 2,048-sample eval subset used during training
  is easier than the full split.

## What broke this morning

Training finished cleanly at 09:07:49 (`[done]`, rc=0) — the shutdown path that
destroyed the previous run worked. Then the pipeline's **holdout eval crashed**:

    RuntimeError: value cannot be converted to type c10::Half without overflow

`play.py`'s `eval_mode` masks illegal moves with `-1e9`, but ran the model under
`torch.autocast`, so the logits were float16 and -1e9 overflows half (max
~65504). `train.py` already cast to float before masking; `play.py` never got
the same fix. A second bug (`.item()` on a Python bool) sat right behind it.

Both fixed and **verified by re-running the exact command that crashed**, plus
a 60-ply game to exercise the move-selection path. No model or data was at risk
— the crash was downstream of every checkpoint.

Honest note: my audit verified the AMP masking in `train.py` and did not carry
that check across to `play.py`. `--mode eval` was the one path never exercised
before the run, so it failed at the end of the pipeline — the same *shape* of
failure as the original bugs, in a file I had checked for other issues.

## The main finding: 22 epochs was too many

From ~epoch 10 the model overfit. Train CE kept falling (1.73 → 1.16) while
held-out val CE rose:

| step | val CE | top1 |
|---|---|---|
| 24,000 | 1.847 | 0.4424 |
| 40,000 | 1.900 | 0.4463 |
| 44,000 | 1.983 | 0.4307 |
| 48,000 | 2.028 | 0.4341 |

The sharpest way to put it: **steps 24,000 → 40,000 (~6 epochs, ~90 min of GPU)
bought +0.66pp top1 while CE got 0.057 worse.** Those epochs were close to
wasted.

**The real lever is more unique positions, not more epochs.** 2.6M records is
~4.7% of one month of Lichess. Next run: either stop at epoch 10-12, or keep 22
epochs and raise the data well past 2.6M. The `d=320 L=8` model (9.92M params,
~40% slower) is the other axis worth an A/B now that there's a baseline.

## Hardening added

- **`--deadline`** (absolute unix timestamp, plumbed through the supervisor and
  `run_pipeline.sh` as `TRAIN_UNTIL`). Nothing previously bounded training in
  wall-clock terms. Reserves 180s to save and eval so it exits via the normal
  `[done]` path. **Tested before the real run** on a synthetic split — stopped
  at step 486, wrote both checkpoints, printed `[done]`, supervisor exited 0.
- **Read-only checkpoint snapshots** at each peak, which is what makes the
  ranking above possible.
- **`compare_checkpoints.py`** — scores every checkpoint on the full split
  instead of trusting the noisy 2,048-sample `best_acc`.
- **README** corrected: 5.58M params (measured), not "~10M".

## One thing left, deliberately not done

`train.py:206` overwrites `ckpt.pt` whenever the final eval beats `best_acc` —
using the same 2,048-sample subset, where 1σ ≈ 1.10pp. At step 38,000 exactly
that happened: a +0.19pp (0.17σ) coin-flip win replaced a checkpoint.

The fix is written and tested in **`apply_ckpt_guard.py`** (requires a >1σ win).
I did **not** apply it mid-run — editing `train.py` under a live trainer risks
the exact class of end-of-run failure this whole audit was about, and the
snapshots already protected this run. Apply it any time:

    python apply_ckpt_guard.py --dry-run
    python apply_ckpt_guard.py

It refuses to run while a `train.py` process is alive.

---

# Scale-up run — 2026-10-01

## Short version

Shipped model replaced: **top1 43.18% → 51.08%** on the same 98,294-position
split (+7.90pp = **49σ**, 1σ = 0.16pp), CE 1.9434 → 1.5452. Both axes the
previous report recommended were taken at once: 4.8× the data and the larger
d=320 L=8 model.

**Ship `/home/aman/Work/tries/modal-september/data2/ckpt_d320L8_12M.pt`**
(step 144,000, best top1). The `.pt.final` sibling is 0.09pp behind = 0.5σ, so
either is defensible; prefer top1 for a move-playing bot.

Also found and fixed a real training bug — `--value-weight 0.5` was ~1000× too
large (below), which had been silently starving the policy head in
`train_eval.py`.

## Data

Source `lichess_db_standard_rated_2023-02`, **first 150.0 GB of the stream only**
(the month's full download measured ~40 MB/s aggregate across parallel fetches,
which would not fit a working session). Stop-on-size means the prefix is a valid
PGN prefix, not a truncated archive.

    150,005,088,256 B decompressed -> 65,417,816 games indexed
    eligible 13,691,948  keep_frac 0.876427460945022  (target 12,000,000)
    train 12,476,546 / val 98,294     860.9 MB / 6.8 MB

Filter unchanged from the shipped pipeline: `min_tc=180s`, `min_clk=10s`,
`clk_frac=0.05` (rejects games that burn their increment). On a sampled shard
that kept 10.4% of games, at 2.01 records per kept game.

Stockfish eval (2000 nodes, side-to-move POV, ±2000 cp clamp), all 12.57M
positions:

    mean 37.1 cp   std 411.9   clamped 251,382 (2.01%)
    legal masks: 407.3M slots, avg 32.6 legal moves/position

## Bug: `value_weight` was ~1000× too large

`train_eval.py` computes `pred_cp = CP_SCALE * tanh(raw)` with `CP_SCALE = 2000`,
then `loss = ce + value_weight * huber(pred_cp, target_cp)`. The `tanh` scale
puts a **2000× multiplier on the value gradient** relative to CE. With the
default `value_weight=0.5` the total gradient is value-dominated, and
`clip_grad_norm_(1.0)` then scales the whole vector down — the policy head
receives almost nothing.

Measured at step 2000 (d=320 L=8, batch 512, eval-n 2048):

| value_weight | lr | ce | top1 | agree |
|---|---|---|---|---|
| 0.5 (was the default) | 3e-4 | 3.1205 | 0.1621 | 0.6431 |
| 0.0005 | 3e-4 | 2.7965 | 0.2212 | 0.7324 |
| 0.0005 | 4e-4 | 2.7607 | 0.2192 | 0.7441 |

At vw=0.5 the run was **flat** — step 10000 gave ce 3.1635 / top1 0.1650, no
better than step 2000. At 0.0005/4e-4 step 4000 already reached ce 2.4004 /
top1 0.3164, matching the old run's step-4000 curve. `0.0005` is what this run
used. lr 3e-4 vs 4e-4 was a tie.

Note `train.py`'s `value_weight` is unaffected — its target is ±1, not ±2000.

## Result

Scored on the **full 98,294-position val split** (1σ on top1 = 0.16pp), with the
two 2.6M checkpoints copied in so all four are judged on identical data:

| checkpoint | step | CE | top1 |
|---|---|---|---|
| shipped best-CE (2.6M, d256 L7) | 24,000 | 1.8829 | 0.4288 |
| shipped `ckpt.pt` (2.6M, d256 L7) | 40,000 | 1.9434 | 0.4318 |
| **new `ckpt_d320L8_12M.pt`** | 144,000 | **1.5452** | **0.5108** |
| new `.pt.final` | 146,208 | 1.5428 | 0.5099 |

**+7.90pp over the shipped model = 49σ.** CE improved by 0.40 nats — the old
run's finding that "steps 24,000 → 40,000 bought +0.66pp while CE got worse" did
not recur here: this run's val CE fell monotonically for all 6 epochs.

Training config: d=320, L=8, 9.92M params, batch 512, lr 4e-4,
value_weight 0.0005, 6 epochs = 146,208 steps, 630 min at ~1,980 samples/s.
Checkpoint gated on `agree` (sign agreement with Stockfish), tie-broken on top1.

## Caveats

- **`val_mse` is not comparable across these rows.** The new head regresses
  Stockfish centipawns; the column still scores it against the game result
  (±1). 0.836 vs 0.761 says the new head disagrees more with who won — expected,
  because Stockfish eval is not the game outcome. `train_eval.py` never reads
  `rows["value"]` for the loss.
- The in-training 8,192-position eval read **0.5040** at step 142,000 where the
  full split says 0.5108 — the subsample was pessimistic by 0.7pp, within its
  0.8pp noise. Quote the full-split number.
- Only **one month's prefix** was used. The 2023-02 remainder is still on the
  Modal volume (`pgn/2023-02.pgn`, 150 GB, plus all 64 prep shards), so a
  larger corpus does not require a re-download.

## Not done

- **No Elo or blunder benchmark.** `estimate_elo.py` and `bench_blunders.py`
  both shell out to Stockfish, which is not installed on this machine and is not
  in the synced pacman db. `+7.9pp top1` is the headline until an engine is
  available.
- The 150 GB PGN and the per-shard `.bin` files were **left on the Modal volume**
  — within the free storage tier, and they let the prep be re-run with different
  filters (different `keep_frac`, clk thresholds) without re-downloading.
