#!/usr/bin/env bash
# Heartbeat + alerting for the overnight pipeline.
# Tolerates brief gaps (a restarted stage) instead of treating them as fatal.
cd "$(dirname "$0")"
missing=0
while true; do
  sleep 600
  if grep -q "pipeline complete" data/pipeline.log 2>/dev/null; then
    echo "PIPELINE FINISHED: $(grep -E 'top1|best|final' data/pipeline.log | tail -3 | tr '\n' ' ')"
    break
  fi
  if pgrep -f "prepare.py" >/dev/null 2>&1; then
    missing=0
    echo "heartbeat prep: $(( $(stat -c %s data/train.bin 2>/dev/null || echo 0) / 69 )) records"
  elif pgrep -f "train.py" >/dev/null 2>&1; then
    missing=0
    echo "heartbeat train: $(grep -E '^step|^  \[eval\]' data/train.log 2>/dev/null | tail -1)"
  elif pgrep -f "run_pipeline.sh" >/dev/null 2>&1; then
    missing=0
    echo "heartbeat: between stages -- $(tail -1 data/pipeline.log)"
  else
    missing=$((missing+1))
    if [ "$missing" -ge 2 ]; then
      echo "ALERT: nothing running for ~20min and pipeline never completed -- last: $(tail -2 data/pipeline.log | tr '\n' ' ')"
      break
    fi
    echo "note: no stage process seen (check $missing/2) -- $(tail -1 data/pipeline.log)"
  fi
done
