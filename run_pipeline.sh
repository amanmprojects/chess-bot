#!/usr/bin/env bash
# Wait for prepare.py to finish, then run masks -> validate -> supervised training.
# Chained so the whole night proceeds unattended. Any stage failing stops the chain
# loudly rather than training on bad data.
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-$HOME/code/llm/.venv/bin/python}"
PREP_PID="${PREP_PID:-}"
LOG=data/pipeline.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -n "$PREP_PID" ]; then
  say "waiting for prepare.py (pid $PREP_PID)"
  while kill -0 "$PREP_PID" 2>/dev/null; do sleep 30; done
  say "prepare.py finished"
fi

if ! grep -q "^\[done\]" data/prep.log; then
  say "FATAL: prep.log has no [done] line -- extraction did not complete cleanly"
  exit 1
fi
say "$(grep '^\[done\]' data/prep.log | tail -1)"

say "=== precomputing masks ==="
"$PY" -u precompute_masks.py --data-dir data >>data/precompute.log 2>&1 || {
  say "FATAL: precompute_masks.py failed"; exit 1; }
say "masks done"

say "=== validating data ==="
"$PY" -u validate_data.py --data-dir data --sample 4000 >data/validate.out 2>&1
vrc=$?
cat data/validate.out | tee -a "$LOG"
if [ "$vrc" -ne 0 ]; then
  say "FATAL: validation failed (rc=$vrc) -- NOT training on suspect data"
  exit 1
fi
say "validation passed"

say "=== training ==="
# Hard wall-clock stop. The user needs a finished, evaluated model by 10:00, so
# training gives up its remaining epochs at TRAIN_UNTIL rather than overrunning.
# Absolute timestamp => resumes cannot extend past it.
TRAIN_UNTIL="${TRAIN_UNTIL:-09:20}"
deadline_ts=$(date -d "today $TRAIN_UNTIL" +%s)
[ "$deadline_ts" -le "$(date +%s)" ] && deadline_ts=$(date -d "tomorrow $TRAIN_UNTIL" +%s)
say "training deadline $(date -d "@$deadline_ts" '+%Y-%m-%d %H:%M') (graceful stop)"
EPOCHS="${EPOCHS:-22}" BATCH="${BATCH:-1024}" LR="${LR:-4e-4}" \
  EVAL_EVERY="${EVAL_EVERY:-2000}" DEADLINE="$deadline_ts" ./supervise_train.sh
rc=$?
say "training exited rc=$rc"

if [ -f data/ckpt.pt ]; then
  say "=== holdout eval ==="
  "$PY" -u play.py --ckpt data/ckpt.pt --mode eval --eval-n 3000 2>&1 | tee -a "$LOG"
fi
say "=== pipeline complete ==="
