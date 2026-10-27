"""CNN + Transformer arbitrage-signal model (PyTorch).

Reimplements the deep model of Guijarro-Ordonez, Pelger & Zanotti (2021):

    input  : the last L cumulative residuals of ONE arbitrage portfolio, x in R^L
    CNN    : local filters capture short patterns (trend, reversal) -> x~ in R^{L x D}
    Transf : multi-head attention captures global dependencies between local
             patterns; the last time step is the signal  theta = hproj_L in R^D
    FFN    : allocation head maps theta -> a scalar weight w^eps in [-1, 1]

Portfolio construction and the Sharpe objective live in ``src/train.py`` — this
module only maps one residual window to one (pre-normalization) weight.

Note on look-ahead: the whole input window ``x`` consists of residuals up to
t-1, all known at decision time. So mixing time steps *inside* the window (as a
non-causal convolution / full attention does) is NOT look-ahead bias — the
no-look-ahead constraint is enforced at the window level by how the data is
built (see train.py), not inside the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvTransformerSignal(nn.Module):
    def __init__(
        self,
        lookback: int = 30,
        n_filters: int = 8,      # D: number of local filters / model width
        kernel_size: int = 2,    # Dsize: local window of the convolution
        n_cnn_layers: int = 2,
        n_heads: int = 4,        # H: number of attention heads (must divide n_filters)
        ffn_hidden: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_filters % n_heads != 0:
            raise ValueError("n_filters must be divisible by n_heads")
        self.lookback = lookback

        # --- CNN: (B, 1, L) -> (B, D, L). Left-pad by kernel_size-1 so the
        # length is preserved without the even-kernel 'same'-padding warning. ---
        conv_layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(n_cnn_layers):
            conv_layers += [
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(in_ch, n_filters, kernel_size),
                nn.InstanceNorm1d(n_filters, affine=True),
                nn.ReLU(),
            ]
            in_ch = n_filters
        self.cnn = nn.Sequential(*conv_layers)

        # --- Transformer: one multi-head self-attention block with residual ---
        self.attn = nn.MultiheadAttention(
            embed_dim=n_filters, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(n_filters)

        # --- Allocation FFN: signal theta (R^D) -> scalar weight in [-1, 1] ---
        self.alloc = nn.Sequential(
            nn.Linear(n_filters, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) cumulative-residual windows -> (B,) raw weights in [-1, 1]."""
        h = x.unsqueeze(1)                 # (B, 1, L)
        h = self.cnn(h)                    # (B, D, L)
        h = h.transpose(1, 2)              # (B, L, D)
        attn_out, _ = self.attn(h, h, h)   # (B, L, D)
        h = self.norm(h + attn_out)        # residual connection + layer norm
        theta = h[:, -1, :]                # (B, D): signal = last-step projection
        w = self.alloc(theta).squeeze(-1)  # (B,)
        return w
