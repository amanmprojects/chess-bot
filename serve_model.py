"""HTTP inference server: serves policy-net moves to the chess web game.

The browser game (chess/ submodule, branch neural-policy) is a zero-dependency
static site whose own AI is alpha-beta search in a Web Worker. This model is a
5.58M-param PyTorch net needing CUDA and a 67MB checkpoint, so it cannot run in
that worker. It runs here instead and the game calls it over HTTP.

    python serve_model.py [--port 8001] [--ckpt data/ckpt.pt]

    POST /move   {"fen": "...", "temperature": 0.0}
              -> {"uci": "e2e4", "value": 0.12, "cp": 118, "ms": 7}
    GET  /health -> {"ok": true, "step": 40000, "device": "cuda"}

Uses only the standard library plus torch, matching the game's no-dependencies
spirit. Single-threaded on purpose: one GPU, one model, and requests are a few
milliseconds each, so a thread pool would only add contention.
"""

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import torch

from model import ChessNet
from play import pick_move

MAX_BODY = 8192          # a FEN is ~90 bytes; anything larger is not a real request


class Handler(BaseHTTPRequestHandler):
    net = None
    device = None
    ckpt_step = None

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        # The page is served by server.mjs on a different port, so every
        # request here is cross-origin. Local dev tool, so allow any origin.
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"ok": True, "step": self.ckpt_step,
                         "device": str(self.device)})

    def do_POST(self):
        if self.path.split("?")[0] != "/move":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": "missing or oversized body"})
            return

        try:
            req = json.loads(self.rfile.read(length))
            fen = req["fen"]
            temperature = float(req.get("temperature", 0.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return

        # A malformed FEN is a client bug, not a server crash.
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            self._send(400, {"error": f"bad fen: {exc}"})
            return

        if board.is_game_over():
            self._send(200, {"uci": None, "reason": "game over",
                             "result": board.result()})
            return

        started = time.perf_counter()
        try:
            move, value = pick_move(self.net, board, self.device, temperature)
        except Exception as exc:
            # pick_move raises if the argmax slot has no legal counterpart.
            # Report it rather than returning a random move: a wrong move here
            # would silently look like the model playing badly.
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        # The value head is a tanh win/draw/loss estimate from the side to
        # move. The game's eval bar wants centipawns from White's view, so
        # convert with the standard logistic mapping and flip for Black.
        v = max(min(float(value), 0.999), -0.999)
        cp = int(-400 * math.log10(2 / (v + 1) - 1))
        if board.turn == chess.BLACK:
            cp = -cp

        self._send(200, {
            "uci": move.uci(),
            "value": round(float(value), 4),
            "cp": cp,
            "ms": int((time.perf_counter() - started) * 1000),
        })

    def log_message(self, fmt, *a):
        # Default logging writes a line per request to stderr; too noisy when
        # the game polls during a match.
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/ckpt.pt")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; this server has no auth")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = ChessNet(d=args.d, n_layers=args.n_layers, n_heads=8).to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    Handler.net = net
    Handler.device = device
    Handler.ckpt_step = int(ck.get("step", -1))

    # Warm up: the first CUDA forward pass pays kernel-compilation cost that
    # would otherwise land on the player's first move as a visible stall.
    pick_move(net, chess.Board(), device, 0.0)

    print(f"model {args.ckpt} (step {Handler.ckpt_step}) on {device}")
    print(f"serving http://{args.host}:{args.port}  (POST /move, GET /health)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
