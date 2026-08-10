#!/usr/bin/env bash
# Relaunch train.py --resume until the run genuinely finishes, so a crash or
# server restart can't silently burn the training budget.
set -uo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-data/ckpt.pt}"
LOG="${LOG:-data/train.log}"
PY="${PY:-$HOME/code/llm/.venv/bin/python}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-1024}"
LR="${LR:-4e-4}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
# Absolute unix timestamp; train.py stops gracefully before it. Passed through
# every resume attempt so a crash-loop cannot extend past the deadline.
DEADLINE="${DEADLINE:-0}"

ckpt_mtime() { [ -f "$CKPT" ] && stat -c %Y "$CKPT" || echo 0; }

fails=0
last_ckpt=$(ckpt_mtime)
while :; do
  echo "== attempt $(date '+%H:%M:%S') ==" | tee -a "$LOG"
  # Capture THIS attempt's output separately: grepping the accumulated log would
  # let a [done] from any earlier run short-circuit every future attempt.
  attempt_log=$(mktemp)
  "$PY" -u train.py --data-dir data --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
        --eval-every "$EVAL_EVERY" --deadline "$DEADLINE" --out "$CKPT" --resume 2>&1 \
        | tee -a "$LOG" >"$attempt_log"
  rc=${PIPESTATUS[0]}
  if grep -q "^\[done\]" "$attempt_log"; then
    rm -f "$attempt_log"
    echo "== [done] training complete ==" | tee -a "$LOG"
    exit 0
  fi
  rm -f "$attempt_log"
  now=$(ckpt_mtime)
  if [ "$now" -gt "$last_ckpt" ]; then
    echo "== exit $rc, checkpoint advanced -- resuming ==" | tee -a "$LOG"
    fails=0
    last_ckpt=$now
  else
    fails=$((fails + 1))
    echo "== exit $rc, no checkpoint progress ($fails failures) ==" | tee -a "$LOG"
    [ "$fails" -ge 20 ] && exit 1
  fi
  sleep 10
done
