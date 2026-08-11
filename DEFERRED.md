# Deferred tasks (do NOT do until told "LATER" / "now")

Recorded 2026-08-10. These are explicitly deferred — the user said "just remember, do not do this now."

1. **Add this repo as a remote and commit nicely.** The chess-bot superproject goes to GitHub
   (`amanmprojects`). Includes: submodule `neural-policy` branch commit + push (browser
   integration, provenance logging), superproject submodule-pointer update, then clean
   well-separated commits. `data/` (670MB) stays gitignored — never commit checkpoints.

2. **Put the model on HuggingFace.** HF token is in `~/code/llm/.env` — read it from there.
   Upload the checkpoint + model definition so the model is recoverable without the local
   data dir.

3. **Wire everything up.** Most likely: `serve_model.py` (or the loading path) can pull the
   model from HF as a fallback. Confirm interpretation when we get here.

Do not act on any of this until the user explicitly tells us to proceed.
