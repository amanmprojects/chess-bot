"""Apply the noise-floor guard to train.py's final-eval overwrite.

Run this ONLY when no training process is using train.py.

Why: train.py's final eval overwrites ckpt.pt whenever its top1 beats best_acc
by any margin -- but both numbers come from the same --eval-n subsample. At
n=2048 and p~0.44 the 1-sigma noise is ~1.10pp, so a coin-flip win can replace a
genuinely better checkpoint with a worse one. That happened on 2026-08-10 at step
38,000: top1 0.4443 beat 0.4424 by +0.19pp = 0.17 sigma.

The guard requires the final eval to win by more than 1 sigma before overwriting.
It is deliberately conservative: ckpt.pt already holds the best model seen during
training, so the burden of proof belongs on the challenger. The final weights are
always available at ckpt.pt.final regardless, so nothing is lost either way.

    python apply_ckpt_guard.py [--dry-run]
"""

import argparse
import math
import sys
from pathlib import Path

OLD = """        if vacc > best_acc:
            best_acc = vacc"""

NEW = """        # Both vacc and best_acc come from the same --eval-n subsample, so a
        # small win is indistinguishable from noise. Require more than 1 sigma
        # before replacing a checkpoint that already proved itself in training.
        noise = math.sqrt(max(vacc, 1e-6) * (1 - vacc) / max(len(idx), 1))
        if vacc > best_acc + noise:
            best_acc = vacc"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    p = Path(__file__).parent / "train.py"
    src = p.read_text()

    if "vacc > best_acc + noise" in src:
        print("already applied -- nothing to do")
        return

    # Refuse to touch a file a live trainer is reading.
    #
    # Do NOT pattern-match command lines: any shell command that merely mentions
    # train.py (including this script's own invoking shell) matches, so pgrep
    # produces false positives that block a legitimate run. Instead ask the
    # kernel which processes actually have the file open, via /proc -- argv
    # cannot spoof that.
    target = p.resolve()
    holders = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            for fd in (pid_dir / "fd").iterdir():
                if fd.resolve() == target:
                    cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ")
                    holders.append(f"{pid_dir.name} {cmd.decode(errors='replace').strip()}")
                    break
        except (PermissionError, FileNotFoundError, OSError):
            continue
    busy = "\n".join(holders)
    if busy:
        sys.exit(f"REFUSING: train.py still running:\n{busy}")

    if src.count(OLD) != 1:
        sys.exit(f"expected exactly 1 match for the overwrite guard, "
                 f"found {src.count(OLD)} -- apply by hand")

    src = src.replace(OLD, NEW)
    if "\nimport math" not in src and "^import math" not in src:
        src = src.replace("import argparse", "import argparse\nimport math", 1)

    if args.dry_run:
        print("would patch train.py:206 (noise-floor guard) and ensure "
              "`import math`")
        return

    p.write_text(src)
    print("patched train.py -- final eval now needs a >1 sigma win to overwrite")
    print(f"(at n=2048, p=0.44 that is {math.sqrt(.44*.56/2048)*100:.2f}pp)")


if __name__ == "__main__":
    main()
