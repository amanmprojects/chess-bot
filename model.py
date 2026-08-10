"""The policy+value transformer: 64 square tokens + 3 aux tokens.

Input:  pieces (B, 64) piece ids, aux (B, 2) [side/castling bits, ep file].
Output: policy logits (B, 4672) in the 64x73 slot space, value (B,) in (-1, 1).
"""

import torch
import torch.nn as nn

NUM_SQUARES = 64
NUM_AUX = 3
SEQ_LEN = NUM_SQUARES + NUM_AUX
POLICY_SIZE = NUM_SQUARES * 73


class Block(nn.Module):
    def __init__(self, d, heads, mlp_scale):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_scale), nn.GELU(),
                                 nn.Linear(d * mlp_scale, d))

    def forward(self, x):
        n = self.ln1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ChessNet(nn.Module):
    def __init__(self, d=256, n_layers=7, n_heads=8, mlp_scale=4):
        super().__init__()
        self.d = d
        self.piece_emb = nn.Embedding(13, d)
        self.side_emb = nn.Embedding(2, d)
        self.castle_emb = nn.Embedding(16, d)
        self.ep_emb = nn.Embedding(16, d)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, mlp_scale) for _ in range(n_layers)])
        self.policy_head = nn.Linear(d, 73)
        self.value_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, pieces, aux):
        B = pieces.shape[0]
        x = self.piece_emb(pieces) + self.pos[:NUM_SQUARES].unsqueeze(0)
        castle = (aux[:, 0] >> 1) & 0xF
        aux_toks = torch.stack(
            [self.side_emb(aux[:, 0] & 1), self.castle_emb(castle),
             self.ep_emb(aux[:, 1])], dim=1) + self.pos[NUM_SQUARES:].unsqueeze(0)
        x = torch.cat([x, aux_toks], dim=1)
        for block in self.blocks:
            x = block(x)
        squares = x[:, :NUM_SQUARES]
        logits = self.policy_head(squares).reshape(B, POLICY_SIZE)
        value = torch.tanh(self.value_head(squares.mean(dim=1)).squeeze(-1))
        return logits, value

    def num_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters()
                   if not trainable_only or p.requires_grad)
