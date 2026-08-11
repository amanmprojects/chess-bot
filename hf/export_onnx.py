"""Export the policy net to ONNX for in-browser inference (onnxruntime-web).

The shipped checkpoint (data/ckpt.pt) is a PyTorch state dict, but a static
HuggingFace Space has no Python process — the model has to run in the browser.
This script exports ChessNet to a single model.onnx and verifies the exported
graph against the PyTorch model on a handful of positions, including the full
argmax-policy move selection used by play.py.

    python hf/export_onnx.py --ckpt data/ckpt.pt --out hf/onnx/model.onnx

Run with the project venv (torch + onnx + onnxruntime).
"""

import argparse
import pathlib
import sys

import chess
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import ChessNet  # noqa: E402
from features import board_to_features, move_to_slot, policy_mask  # noqa: E402

D = 256
N_LAYERS = 7
N_HEADS = 8
OPSET = 17  # Gelu is a native op from here; wide onnxruntime-web support


def export(ckpt_path: pathlib.Path, out_path: pathlib.Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = ChessNet(d=D, n_layers=N_LAYERS, n_heads=N_HEADS)
    net.load_state_dict(ck["model"])
    net.eval()

    pieces = torch.zeros(1, 64, dtype=torch.long)
    aux = torch.zeros(1, 2, dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (pieces, aux),
        str(out_path),
        input_names=["pieces", "aux"],
        output_names=["logits", "value"],
        dynamic_axes={"pieces": {0: "batch"}, "aux": {0: "batch"}},
        opset_version=OPSET,
    )
    # The dynamo exporter spills weights into a sidecar model.onnx.data; a
    # static Space wants one self-contained file, so inline them.
    import onnx

    m = onnx.load(str(out_path))
    onnx.save(m, str(out_path), save_as_external_data=False)
    onnx.checker.check_model(m)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes")

    # --- Verify against torch on a spread of positions ---------------------
    import onnxruntime as ort

    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/8/8/8/8/2k5/1p6/1K6 b - - 0 1",          # promotion
        "8/4P3/8/8/8/8/8/k1K5 w - - 0 1",            # queen + underpromotions
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",      # castling rights
    ]
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst_logit = worst_value = 0.0
    for fen in fens:
        board = chess.Board(fen)
        pieces_np, aux_np = board_to_features(board)
        p = torch.from_numpy(pieces_np).long().unsqueeze(0)
        a = torch.from_numpy(aux_np).long().unsqueeze(0)
        with torch.no_grad():
            logits_t, value_t = net(p, a)

        feeds = {
            "pieces": pieces_np.astype(np.int64)[None],
            "aux": aux_np.astype(np.int64)[None],
        }
        logits_o, value_o = sess.run(None, feeds)
        worst_logit = max(worst_logit, float(np.abs(logits_t.numpy() - logits_o).max()))
        worst_value = max(worst_value, float(np.abs(value_t.item() - value_o[0])))

        # The full decision path: mask illegal slots, argmax, find the move.
        mask = policy_mask(board)
        slot_t = int((logits_t[0].float() +
                      torch.where(torch.from_numpy(mask),
                                  torch.zeros(4672),
                                  torch.full((4672,), -1e9))).argmax())
        slot_o = int(np.argmax(logits_o[0] + np.where(mask, 0.0, -1e9)))
        move_t = next(m for m in board.legal_moves if move_to_slot(board, m) == slot_t)
        move_o = next(m for m in board.legal_moves if move_to_slot(board, m) == slot_o)
        status = "ok " if move_t == move_o else "DIFF"
        print(f"  {status} {fen:60s} -> {move_t.uci()} (slot {slot_t} vs {slot_o})")

    assert worst_logit < 1e-4 and worst_value < 1e-4, (worst_logit, worst_value)
    print(f"torch vs onnx: max |Δlogits| = {worst_logit:.2e}, "
          f"max |Δvalue| = {worst_value:.2e}  PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data/ckpt.pt"))
    ap.add_argument("--out", default=str(ROOT / "hf/onnx/model.onnx"))
    args = ap.parse_args()
    export(pathlib.Path(args.ckpt), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
