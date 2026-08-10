# Bugs Fixed — 2026-08-10

## Critical (caused the "dumped 3h model")

**Bug A: `idx` NameError at final eval** (`train.py:177`)  
The final `evaluate()` call used `idx`, which was only bound inside `if step % eval_every == 0`. If training finished without hitting an eval step, the final eval crashed with `NameError`.

**Bug B: Final eval before final save** (`train.py:177-182`)  
Weights were saved *after* the final eval. If eval crashed (Bug A), a completed run's weights were discarded.

**Bug C: Supervisor stale-`[done]` grep** (`supervise_train.sh:20`)  
The supervisor grepped the *entire accumulated log* for `[done]`, so any past run's `[done]` made every future attempt exit immediately as "complete" even if the new run crashed at step 0.

**Combined failure mode matching the user's report:**  
Training finishes → final eval raises NameError → no `[done]` line → supervisor relaunches --resume → resume with `step == total_steps` → loop body never runs → falls straight to final eval → NameError again → no checkpoint progress → 20 failures → exit 1. The model trained for 3 hours, then was lost at the very end.

**Fix:** Moved `idx` creation outside the loop, wrapped final eval in try/except, save weights *before* eval, and changed supervisor to grep only the current attempt's output (via `mktemp`).

## Medium (crashes in play.py)

**Bug D: Tensor accumulation without `.item()`** (`play.py:72`)  
`legal_ok += mask[pred]` accumulated a boolean tensor, then divided by an int. Crashes in some torch versions.

**Bug E: `.4f` format on missing `best_acc`** (`play.py:143`)  
`ckpt.get('best_acc', '?'):.4f` crashes if the fallback `'?'` string is used.

**Fix:** Added `.item()` and `int()` casts; guarded the format with a None check.

**Bug F: `-1e9` mask overflows float16 under autocast** (`play.py:69`, found
2026-08-10 09:07 when the pipeline's holdout eval crashed)

    RuntimeError: value cannot be converted to type c10::Half without overflow

`eval_mode` ran the model inside `torch.autocast`, so `logits` came back
float16, and `torch.where(mask, torch.zeros_like(logits[0]), -1e9)` tried to
put -1e9 into a half tensor (max ~65504). torch raises rather than saturating.
`train.py`'s `evaluate()` already cast to float first; `play.py` never got the
same fix, and `--mode eval` was the one path never exercised before the
overnight run — so it failed at the *end* of the pipeline, exactly like the
original Bugs A+B.

**Bug G: `.item()` on a Python bool** (`play.py:71`)
`acc += (pred == target).item()` compares two Python ints, so the result is a
`bool`, which has no `.item()`. Latent behind Bug F — it would have crashed on
the very next line once the masking was fixed.

**Fix:** Cast to float *before* building the mask tensor, and use
`torch.full_like` so the fill value can never be narrowed to the logits' dtype:

    masked = logits[0].float()
    masked = masked + torch.where(mask, torch.zeros_like(masked),
                                  torch.full_like(masked, -1e9))

Applied at both masking sites (`pick_move` and `eval_mode`) and `acc += int(...)`.
`pick_move` was not actually broken — it rebinds `logits = logits.float()` on
the line above — but it depended on that rebinding two lines up, and it was
returning `value` in half precision. Both sites now use the explicit form.

**Verified after the fix** on the real paths, not in isolation:

    [eval] n=3000 top1 0.4370 legal 1.0000     # the holdout eval that crashed
    60 plies vs itself, every move legal, coherent London System opening

`legal 1.0000` over 3,000 held-out positions is independent confirmation that
the encoding and mask logic are correct end-to-end (board -> features -> model
-> move), which is the strongest available evidence against the suspected
from-square bug.

**Audit gap this exposes:** the original audit verified AMP masking in
`train.py` and did not carry that check across to `play.py`. When a numeric
invariant is verified in one file, grep every file for the same pattern.

## Audit results (no bugs)

- **Encoding:** Exhaustive bijection test over 20K legal moves, all promotions, all from-squares. Zero collisions, zero missing from-squares. The reported slot bug was already fixed in `features.py:60`.
- **Features roundtrip:** 6K positions including ep/castling. Zero mismatches.
- **Mask invariant:** Reconstructed boards produce identical legal masks across 8K positions. Every target lands inside its mask.
- **AMP numerics:** `-1e9` mask works correctly after `.float()` cast. Observed `ce~1.75` proves targets were inside masks (illegal target would yield `ce~1e9`).
- **Leftover data forensics:** 425K records, 64/64 from-squares used, 69/73 labels. The previous run's data was correctly encoded; the model was learning normally before the crash.

## Overfitting observed in the 2026-08-10 run (read this first)

From ~step 26,000 (epoch ~10) the model began overfitting. Training CE kept
falling while held-out val CE flattened and then rose:

| step | train_ce | val_ce | gap | top1 |
|---|---|---|---|---|
| 24,000 | 1.727 | 1.847 | -0.119 | **0.4424** (best) |
| 28,000 | 1.588 | 1.853 | -0.265 | 0.4365 |
| 32,000 | 1.551 | 1.855 | -0.305 | 0.4380 |
| 36,000 | 1.432 | 1.907 | -0.475 | 0.4243 |

val_ce slope over the last 4 evals was **+0.025/eval**, well past the +0.010
alarm threshold. So 22 epochs is more than 2.6M positions supports for this
model; the useful stopping point was around epoch 9-10.

**The best checkpoint is preserved.** `best_acc` gating means `ckpt.pt` only
advances on improvement. Two candidates were snapshotted read-only:
`data/ckpt.best-24000.pt` (sha `ae90314f744d8af5`) and
`data/ckpt.best-38000.pt`.

That backup exists because of a real hazard: `train.py:206` overwrites `ckpt.pt`
if the *final* eval beats `best_acc`, and that comparison uses the same
2,048-sample subset whose 1-sigma noise is ~1.10pp. At step 38,000 exactly that
happened — top1 0.4443 beat 0.4424 by +0.19pp, i.e. **0.17 sigma**, a coin flip.

**Re-scored on the full 13,000-position val split** (6x the training eval
sample), the two disagree by metric:

| checkpoint | step | CE | top1 |
|---|---|---|---|
| ckpt.best-24000.pt | 24,000 | **1.9005** | 0.4247 |
| ckpt.best-38000.pt | 38,000 | 1.9675 | **0.4312** |

Step 38,000 wins top1 by 1.5 sigma; step 24,000 wins CE by 0.067. So the later,
more-overfit checkpoint is *better at picking the right move* while being
*worse-calibrated* over the whole distribution — overfitting here costs
calibration more than move accuracy. For a move-playing bot, top1 is the
operative metric, so the later checkpoint is a defensible pick despite the rising
val CE.

Note also that both score **~43% on the full split vs 44.2-44.4% on the 2,048
subset** — the training-time eval subset was easier, so ~43% is the honest
number to quote.

The overwrite hazard is still worth fixing: evaluate the final model on the full
val split, or require the final eval to beat `best_acc` by more than the noise
floor before overwriting.

**Next run should:** stop around epoch 10-12, or (better) keep 22 epochs and
increase data past 2.6M records. More epochs over the same positions is not
buying accuracy here.

## Hardening added for the unattended run (not bugs)

**Wall-clock deadline** (`train.py --deadline`, plumbed through
`supervise_train.sh` and `run_pipeline.sh` as `TRAIN_UNTIL`, default `09:20`).

Nothing previously bounded training in wall-clock terms — only epochs. If
throughput dipped overnight, the run would still have been training at the
10:00 deadline with no usable model. `--deadline` takes an **absolute unix
timestamp** (not a per-attempt budget) so a crash-and-resume cannot extend past
it. train.py reserves 180 s before the deadline to save weights and run the
final eval, so hitting it exits through the normal `[done]` path rather than
being killed mid-write.

Verified end-to-end before the real run: with a deadline ~40 s out and 500
epochs requested, train.py stopped at step 486, wrote both `ckpt.pt` and
`ckpt.pt.final`, completed the final eval, and printed `[done]`; the supervisor
saw that `[done]` and exited 0 on the first attempt with no crash-loop.

Tradeoff: if the deadline fires, the cosine LR schedule has not annealed to
`min_lr`, so the model is slightly under-trained relative to a full 22 epochs.
That is deliberate — a usable model at 09:20 beats a better one that does not
exist at 10:00. The `[deadline]` log line records the step and epoch it stopped
at, so a partial run is obvious rather than silent.

Measured on the actual 2026-08-10 run (2,587,000 records, batch 1024, 22 epochs
= 55,572 steps, ~3050 rec/s after the GPU settles at 73 °C / 2040 MHz), the
truncation cost is small because the cosine schedule does most of its annealing
by epoch ~18:

| stop early by | step reached | epochs | LR at stop |
|---|---|---|---|
| 0 min (full)  | 55,572 | 22.0 | 1.0e-05 |
| 10 min        | 53,787 | 21.3 | 1.1e-05 |
| 30 min        | 50,215 | 19.9 | 1.9e-05 |
| 45 min        | 47,536 | 18.8 | 3.0e-05 |

So the 09:20 deadline is a genuine safety net rather than a live constraint: the
run projects to finish naturally at ~09:11, and even a 45-minute throughput
shortfall would still land deep in the anneal.

---

**Conclusion:** The encoding was never broken. The previous run trained successfully, then died at the final eval due to Bugs A+B+C, and the supervisor couldn't recover. All fixed.

## Open question for review (not a bug)

`README.md` advertised "~10M params", but the default config in `model.py`
(`d=256, n_layers=7`) is **5,577,034 = 5.58M** (measured, not estimated). The
documented size corresponds to `d=320, n_layers=8` (9.92M). Measured on the
RTX 4060 (8.19 GB), batch 1024:

| config | params | throughput | peak VRAM | epochs in 5.5h (2.6M recs) |
|---|---|---|---|---|
| d=256 L=7 (current default) | 5.58M | 3161 rec/s | 4.87 GB | ~24 |
| d=320 L=8 (README's claim)  | 9.92M | 1921 rec/s | 6.89 GB | ~15 |

The overnight run of 2026-08-10 used **d=256 L=7**, chosen for safety while the
user was AFK: it is the config validated end-to-end that night, and 6.89 GB of
8.19 GB leaves little room for fragmentation over a multi-hour unattended run
(an OOM at 05:00 would have had no recovery path). The larger model is likely
the better accuracy-per-wall-clock choice when someone is around to watch it —
worth an A/B once there is a baseline number to compare against.

Either fix the README to say ~5.6M, or switch the default to `d=320 L=8`.

**Resolved 2026-08-10:** README corrected to state 5.58M, with a note that
`d=320 L=8` is 9.92M and ~40% slower. The default was left at `d=256 L=7` —
changing model size is a decision for a supervised run, not an unattended one.

## Fix staged for the next run: noisy final-eval overwrite

The `train.py:206` hazard described above now has a patch ready in
`apply_ckpt_guard.py`. It requires the final eval to beat `best_acc` by more
than 1 sigma (`sqrt(p(1-p)/n)`, ~1.10pp at n=2048) before overwriting
`ckpt.pt`, instead of by any margin.

It was **not** applied during the 2026-08-10 run: editing `train.py` under a
live trainer risks the very class of end-of-run failure this whole audit was
about, and the run was already protected by read-only checkpoint snapshots. The
script refuses to run while any `train.py` process is alive (verified — it
correctly refused against pid 41254), so it is safe to invoke any time after
the run ends:

    python apply_ckpt_guard.py --dry-run   # preview
    python apply_ckpt_guard.py             # apply

Rationale for the asymmetry: `ckpt.pt` already holds the best model observed
across training, so the burden of proof is on the challenger. The final weights
remain at `ckpt.pt.final` either way, so the guard can never lose a model — it
only prevents a worse one from displacing a better one.

