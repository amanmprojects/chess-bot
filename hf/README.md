---
license: mit
library_name: pytorch
pipeline_tag: other
tags:
  - chess
  - policy-network
  - transformer
  - alphazero
---

# chess-policy-net

A 5.58M-parameter policy+value transformer that plays chess from a single
forward pass — no search. Trained by supervised learning on human games from
the January 2023 Lichess dump.

It picks moves the way a strong-ish club player picks them *before* calculating:
by pattern. Given a position it emits a distribution over the 4,672-slot move
space, illegal slots are masked out, and the argmax is played. One position is
examined per move, so it has no tactical lookahead at all — that is the main
thing to know about its playing strength.

## Results

Scored on a held-out 13,000-position split (1σ on top1 ≈ 0.43pp):

| metric | value |
|---|---|
| top-1 move agreement | **43.1%** |
| top-1, independent 3,000-position holdout | 43.7% |
| legal-move rate | **100.0%** |
| cross-entropy | 1.957 |
| value head MSE | 0.75 |

Playing strength against Stockfish at fixed Skill Levels, 24–30 games each,
fast time control:

| opponent | score | est. Elo | 95% CI |
|---|---|---|---|
| Stockfish ≈1320 | 8.5/30 (28%) | 1159 | 1024–1293 |
| Stockfish ≈1400 | 7.5/30 (25%) | 1209 | 1069–1349 |
| Stockfish ≈1500 | 1.0/30 (3%) | 915 | 610–1220 |

**Treat this as a band of roughly 1150–1200 Elo, not a rating.** The samples are
small, the anchors are approximate, and a searchless engine's results are
noisier than the CIs suggest. The sharp fall-off at 1500 is what a
no-lookahead model looks like when the opponent starts setting two-move traps.

## Files

| file | what it is |
|---|---|
| `ckpt.pt` | the checkpoint (67MB) — weights, AdamW state, step, best_acc |
| `model.py` | `ChessNet` definition |
| `features.py` | board encoding and the 64×73 move-slot mapping |
| `model.onnx` | the same weights exported to ONNX for in-browser play |

`ckpt.pt` keeps the optimizer moments so training can be resumed from it, which
is why it is 67MB rather than the ~22MB the weights alone need. `model.onnx`
is regenerated from `ckpt.pt` by `hf/export_onnx.py` in the source repo and is
verified to produce identical moves (max |Δlogits| < 1e-4).

## Play it in the browser

The chess UI is a static HuggingFace Space — no Python involved. `model.onnx`
runs inside a Web Worker via onnxruntime-web (WebAssembly):

**[huggingface.co/spaces/amanm10000/chess-bot](https://huggingface.co/spaces/amanm10000/chess-bot)**

Pick **Neural net** in the level dropdown. The feature encoding and move-slot
mapping are ported to JS in the Space's `src/neural.js`, pinned by tests
against python-chess fixtures.

## Usage

Needs `torch` and `python-chess`. `features.py` is not optional — it holds the
exact encoding the weights were trained against, and any mismatch produces
legal but meaningless moves.

```python
import chess, torch
from huggingface_hub import hf_hub_download

from model import ChessNet          # from this repo
from features import board_to_features, policy_mask

path = hf_hub_download("amanm10000/chess-policy-net", "ckpt.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

net = ChessNet(d=256, n_layers=7, n_heads=8)
net.load_state_dict(ckpt["model"])
net.eval()

board = chess.Board()
pieces, aux = board_to_features(board)
with torch.no_grad():
    logits, value = net(torch.from_numpy(pieces).long()[None],
                        torch.from_numpy(aux).long()[None])

# Mask illegal slots before the argmax. Keep this in float32: -1e9 overflows
# float16, which torch raises on rather than saturating.
mask = torch.from_numpy(policy_mask(board))
masked = logits[0].float().masked_fill(~mask, -1e9)
slot = int(masked.argmax())
print(slot, float(value))
```

To turn a slot back into a move, scan the legal moves for the one whose
`move_to_slot(board, move)` matches — the mapping is a bijection over legal
moves, so exactly one matches.

## Architecture

```
64 square tokens + 3 aux tokens (side to move, castling rights, en-passant file)
67 learned position vectors
7 pre-norm transformer blocks, d=256, 8 heads, MLP ×4
policy head: shared Linear(d -> 73) per square -> 4672 logits
value head:  mean-pool over squares -> Linear -> tanh
```

**Move encoding** (73 labels per from-square, AlphaZero-style): 56 queen slides
(8 directions × 7 distances), 8 knight jumps, 9 underpromotions (3 destination
files × N/B/R). Castling is a king slide of 2, en passant is a pawn diagonal of
1, and queen promotion is an ordinary slide — so those need no special labels.

**Board encoding**: one uint8 piece id per square (0 empty, 1–6 white PNBRQK,
7–12 black), row-major from a8 to h1.

## Training

| | |
|---|---|
| data | `lichess_db_standard_rated_2023-01`, 9.1M games → 2.6M positions |
| filters | both players ≥1800 Elo, base time ≥180s, Termination=Normal, 12–250 moves |
| split | 2,587,000 train / 13,000 val |
| optimizer | AdamW, cosine LR, AMP, batch 1024 |
| hardware | one RTX 4060 |
| shipped step | 40,000 (~16 epochs) of 55,572 trained |

**The final checkpoint is not the shipped one.** Training ran 22 epochs and the
model overfit from roughly epoch 10 onward: train CE fell 1.73 → 1.16 while
held-out val CE rose 1.847 → 2.16. Step 40,000 is the peak; the step-55,572
final weights score 1.6pp worse. `best_acc` gating is the only reason the good
weights survived.

Steps 24,000 → 40,000 bought +0.66pp top-1 while CE got 0.057 *worse* — about
90 minutes of GPU time for nearly nothing. On this architecture the real lever
is more unique positions, not more epochs: 2.6M records is only ~4.7% of a
single month of Lichess.

## Limitations

- **No search.** It will hang pieces to any two-move tactic. Pairing the policy
  with even shallow alpha-beta would be worth more than any amount of extra
  training at this scale.
- **Imitation, not strength.** The objective is "what would a 1800+ human play
  here", so it inherits human habits, including bad ones, and cannot exceed the
  data.
- **The value head is weak** (MSE 0.75 against ±1 targets). It is useful as an
  eval-bar hint and little else.
- Trained only on standard chess from one month of one site. No Chess960, no
  endgame tablebases, no opening book.

## Source

Training pipeline, evaluation, and the Elo harness:
[github.com/amanmprojects/chess-bot](https://github.com/amanmprojects/chess-bot)
