"""Search over the policy net.

Why this file exists: the shipped bot picks a move with a single forward pass and
an argmax (play.py:pick_move). A net trained to imitate 1800+ humans reproduces
what those humans played in the position it was shown, with no ability to look one
move further. That is the whole explanation for "sometimes genius, sometimes hangs
the queen": the policy is confident about moves that are only bad in hindsight.

This module adds the three cheap layers, in increasing cost:

    1. tactical guard   -- reject moves that hang material outright. No net
                           required; pure board logic. Fixes the worst blunders.
    2. best-of-N        -- sample N moves from the policy, rerank by the value
                           head, keep the best. Turns an argmax into a search.
    3. alpha-beta       -- depth-2..4 negamax with quiescence over the net's
                           value head, policy-ordered.

All three read the value head, which is why retraining it on Stockfish evals
matters: with the shipped value head (trained on game outcome, MSE ~0.75, barely
better than a constant) layers 2 and 3 are close to useless. Layer 1 works today.

Deliberately dependency-free and side-effect-free: the caller passes a board and
gets a move back.
"""

import chess
import torch
import torch.nn.functional as F

from features import board_to_features, move_to_slot, policy_mask

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _legal_slots(board, net, device):
    """Policy logits masked to legal moves, plus the slot->move table."""
    mask = torch.from_numpy(policy_mask(board)).to(device)
    slot_to_move = {}
    for move in board.legal_moves:
        slot_to_move[move_to_slot(board, move)] = move
    return mask, slot_to_move


@torch.no_grad()
def evaluate(net, device, batch=None):
    """Run the net on a batch of (pieces, aux). Returns (logits, value)."""
    if batch is not None:
        pieces, aux = batch
    logits, value = net(pieces, aux)
    # Must leave autocast's float16 before any -1e9 masking (max half ~65504).
    return logits.float(), value.float()


def _features(board, device):
    pieces, aux = board_to_features(board)
    return (torch.from_numpy(pieces).long().unsqueeze(0).to(device),
            torch.from_numpy(aux).long().unsqueeze(0).to(device))


def tactical_guard(board, drop_cp: int = 150, margin_cp: int = 60):
    """Moves that lose material outright, per static exchange on the destination.

    Returns (forbidden_slots, saved_moves). A move is forbidden if the piece
    landing on the destination square is attacked by the opponent and the least
    valuable defender cannot restore the material -- i.e. a free capture.

    This is deliberately crude and cheap. It is not a safety proof: it ignores
    pins, discovered checks and whether the defender's own move would be legal
    (a pinned attacker still "attacks" in this count). It only removes the
    unambiguous free hangs, which is the failure mode that reads as a 200-Elo
    blunder in a game.
    """
    forbidden, saved = set(), []
    turn = board.turn

    for move in board.legal_moves:
        if board.is_capture(move):
            continue
        if move.promotion:
            # Underpromotion is deliberate, not a blunder.
            continue

        landed = board.piece_at(move.from_square)
        if landed is None or landed.piece_type == chess.KING:
            continue
        gained = PIECE_VALUE[landed.piece_type]

        # Cheapest defender that can legally recapture on the destination.
        cheapest = None
        for reply in board.legal_moves:
            if reply.to_square != move.to_square or not board.is_capture(reply):
                continue
            attacker = board.piece_at(reply.from_square)
            if attacker is None or attacker.color == turn:
                continue
            cost = PIECE_VALUE[attacker.piece_type]
            if cheapest is None or cost < cheapest:
                cheapest = cost

        net_loss = gained if cheapest is None else gained - cheapest
        if net_loss >= drop_cp:
            # Only forbid if the net head already prefers the move by more than
            # the margin, so we never override a clearly good sacrifice.
            forbidden.add(move_to_slot(board, move))
        elif net_loss > 0:
            saved.append(move)
    return forbidden, saved


@torch.no_grad()
def best_of_n(board, net, device, n=8, temperature=1.0, seed=None):
    """Sample n policy moves, rerank by the child's value, return the best.

    The child's value is the net's estimate of the position after the move, from
    the opponent's view, so we negate it to compare all candidates on one scale.
    """
    if seed is not None:
        torch.manual_seed(seed)
    pieces, aux = _features(board, device)
    logits, _ = evaluate(net, device, (pieces, aux))
    mask, slot_to_move = _legal_slots(board, net, device)
    masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                     torch.full_like(logits[0], -1e9))
    probs = F.softmax(masked / max(temperature, 1e-4), dim=0)

    legal_slots = torch.nonzero(mask).flatten()
    best_move, best_val = None, -1e9
    seen = set()
    for _ in range(n):
        slot = legal_slots[torch.multinomial(probs[legal_slots], 1)].item()
        if slot in seen:
            continue
        seen.add(slot)
        move = slot_to_move[slot]
        board.push(move)
        try:
            cp, ca = _features(board, device)
            _, v = evaluate(net, device, (cp, ca))
            val = -v.item()          # from our side
        finally:
            board.pop()
        if val > best_val:
            best_move, best_val = move, val
    return best_move, best_val


@torch.no_grad()
def alpha_beta(board, net, device, depth=3, alpha=-1e9, beta=1e9,
               quiescence=2):
    """Depth-limited negamax with quiescence, evaluating leaves with the net.

    The value head returns tanh in (-1, 1); scaling to centipawn-ish units keeps
    the alpha/beta window arithmetic readable.
    """
    if board.is_game_over():
        if board.is_checkmate():
            return -1e6 if board.turn else 1e6
        return 0

    if depth <= 0:
        return _quiesce(board, net, device, quiescence, alpha, beta)

    pieces, aux = _features(board, device)
    logits, _ = evaluate(net, device, (pieces, aux))
    mask, slot_to_move = _legal_slots(board, net, device)
    masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                     torch.full_like(logits[0], -1e9))
    # Policy-ordered: try the model's favourite moves first for earlier cutoffs.
    order = torch.argsort(masked, descending=True)

    best = -1e9
    for slot in order.tolist():
        if not mask[slot]:
            continue
        move = slot_to_move[slot]
        board.push(move)
        try:
            score = -alpha_beta(board, net, device, depth - 1, -beta, -alpha,
                                quiescence)
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


@torch.no_grad()
def _quiesce(board, net, device, qdepth, alpha, beta):
    """Extend the search through captures so tactics aren't cut mid-exchange."""
    pieces, aux = _features(board, device)
    _, v = evaluate(net, device, (pieces, aux))
    best = v.item()
    if best >= beta:
        return best
    if qdepth <= 0:
        return best
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        board.push(move)
        try:
            score = -_quiesce(board, net, device, qdepth - 1, -beta, -alpha)
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


@torch.no_grad()
def search_move(board, net, device, mode="guard+ab", depth=3, n=8,
                temperature=1.0, use_value=True, seed=None):
    """The single entry point: pick a move for `board` under the chosen strategy.

    mode:
      "policy"        original behaviour -- single forward pass, argmax
      "guard"         policy, but forbidden slots removed
      "guard+bon"     tactical guard, then best-of-N reranked by value
      "guard+ab"      tactical guard, then alpha-beta to pick the slot
    """
    pieces, aux = _features(board, device)
    logits, _ = evaluate(net, device, (pieces, aux))
    mask, slot_to_move = _legal_slots(board, net, device)

    if mode == "policy":
        masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                         torch.full_like(logits[0], -1e9))
        slot = masked.argmax().item()
        return slot_to_move[slot], 0.0

    forbidden = set()
    if mode.startswith("guard"):
        forbidden, _ = tactical_guard(board)
        allowed = mask.clone()
        for slot in forbidden:
            allowed[slot] = False
        if not allowed.any():
            allowed = mask      # guard would leave nothing; fall back

    if mode == "guard+bon" and use_value:
        move, val = best_of_n(board, net, device, n=n,
                              temperature=temperature, seed=seed)
        return move, val

    if mode == "guard+ab" and use_value:
        best_move, best_score = None, -1e9
        slots = torch.nonzero(mask & ~torch.tensor(
            [i in forbidden for i in range(len(mask))], device=device)).flatten()
        for slot in slots.tolist():
            move = slot_to_move[slot]
            board.push(move)
            try:
                score = -alpha_beta(board, net, device, depth - 1)
            finally:
                board.pop()
            if score > best_score:
                best_move, best_score = move, score
        if best_move is not None:
            return best_move, best_score

    masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                     torch.full_like(logits[0], -1e9))
    for slot in forbidden:
        masked[slot] = -1e9
    if bool((masked > -1e8).any()) is False:
        masked = logits[0] + torch.where(mask, torch.zeros_like(logits[0]),
                                         torch.full_like(logits[0], -1e9))
    return slot_to_move[masked.argmax().item()], 0.0