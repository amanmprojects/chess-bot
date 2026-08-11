"""Publish the trained policy net to the HuggingFace Hub.

The point is recoverability: `data/` is gitignored (670MB of PGN shards and
checkpoints), so without this the weights exist on exactly one disk. What gets
uploaded is the minimum set that makes the model usable standalone -- the
checkpoint plus the two modules that define what its numbers mean.

`features.py` is not optional. It holds the frozen board encoding and the
64x73 move-slot mapping the weights were trained against; a mismatch there
produces legal but meaningless moves rather than an error.

    python push_to_hf.py --dry-run     # show what would be sent
    python push_to_hf.py
"""

import argparse
import os
import pathlib
import re
import sys

REPO_ID = "amanm10000/chess-policy-net"
ROOT = pathlib.Path(__file__).resolve().parent
ENV_FILE = pathlib.Path.home() / "code/llm/.env"

# (local path, path in the HF repo)
UPLOADS = [
    (ROOT / "data/ckpt.pt", "ckpt.pt"),
    (ROOT / "model.py", "model.py"),
    (ROOT / "features.py", "features.py"),
    (ROOT / "hf/README.md", "README.md"),
    # ONNX export for in-browser play (static Spaces have no Python).
    # Regenerate with: ~/code/llm/.venv/bin/python hf/export_onnx.py
    (ROOT / "hf/onnx/model.onnx", "model.onnx"),
]


def read_token():
    """HF_TOKEN from the environment, else from ~/code/llm/.env."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "environment"
    if not ENV_FILE.exists():
        sys.exit(f"no HF_TOKEN in the environment and no {ENV_FILE}")
    for line in ENV_FILE.read_text().splitlines():
        m = re.match(r"""\s*(?:export\s+)?HF_TOKEN\s*=\s*["']?([^"'\s]+)""", line)
        if m:
            return m.group(1), str(ENV_FILE)
    sys.exit(f"no HF_TOKEN line found in {ENV_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files and exit without contacting the Hub")
    ap.add_argument("--repo", default=REPO_ID)
    args = ap.parse_args()

    missing = [str(p) for p, _ in UPLOADS if not p.exists()]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))

    total = sum(p.stat().st_size for p, _ in UPLOADS)
    print(f"repo {args.repo}")
    for path, dest in UPLOADS:
        print(f"  {path.stat().st_size / 1e6:8.2f} MB  {path.name:12s} -> {dest}")
    print(f"  {total / 1e6:8.2f} MB  total")

    if args.dry_run:
        print("\ndry run, nothing sent")
        return

    token, source = read_token()
    print(f"\ntoken from {source}")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"authenticated as {api.whoami()['name']}")

    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for path, dest in UPLOADS:
        print(f"uploading {dest} ...", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                        repo_id=args.repo, repo_type="model")

    files = sorted(api.list_repo_files(args.repo))
    print(f"\nhttps://huggingface.co/{args.repo}")
    print("files now in the repo:", ", ".join(files))


if __name__ == "__main__":
    main()
