"""Predictor (prediction head) for the JEPA Barlow Twins objective.

Sits on top of the sequence encoder in seq_encoder.py. It maps the sequence encoder's
per-residue representation to a prediction of the MAP encoder's per-residue
representation. Barlow Twins compares the two matrices element-for-element, so:

  * the output stays a per-residue MATRIX (B, L, OUTPUT_DIM) -- no pooling, no
    collapsing to a single vector, and
  * OUTPUT_DIM MUST equal the map encoder's output dimension.

The 6-block sequence encoder already mixed information across residues, so the v1
predictor is deliberately LIGHT: a per-residue MLP applied independently at every
position. It's a translator, not a second encoder.
"""

import torch
import torch.nn as nn

# Reuse the encoder stack and data plumbing from the seq_encoder module. TransformerBlock
# is imported only for the optional heavier variant described at the bottom.
from seq_encoder import (
    TokenEmbedding,
    RoPEEncoder,
    TransformerBlock,
    make_dataloader,
    BATCH_SIZE,
)
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Constants (named, easy to change)
# ---------------------------------------------------------------------------
INPUT_DIM = 512
HIDDEN_DIM = 1024
OUTPUT_DIM = 512      # MUST match the map encoder's output dim (Barlow Twins
                      # compares the two representations element-for-element).
DROPOUT = 0.1


# ---------------------------------------------------------------------------
# Predictor: per-residue MLP, applied independently at every position
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    """LayerNorm -> Linear(512->1024) -> GELU -> Dropout -> Linear(1024->OUTPUT_DIM).

    A plain nn.Linear acts over the last dim only, so its weights are shared
    across all L positions and it handles any sequence length automatically.

    NOTE: this head is purely per-residue -- there is no attention or any other
    cross-position mixing here. Position i's output depends only on position i's
    input, so padding positions cannot leak into real positions. No masking is
    therefore needed INSIDE the predictor; the mask is only threaded through so
    the downstream Barlow Twins loss can drop padding positions.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask):
        """x: (B, L, INPUT_DIM), mask: (B, L) -> (predicted (B, L, OUTPUT_DIM), mask)."""
        return self.net(x), mask


# ---------------------------------------------------------------------------
# OPTIONAL heavier variant -- enable only if the per-residue MLP underfits.
# ---------------------------------------------------------------------------
# If the v1 MLP can't fit the map representation, prepend a few narrow pre-LN
# transformer blocks (reusing seq_encoder.TransformerBlock) before the final Linear.
# Because those blocks DO use attention, the padding mask MUST be passed into
# them -- otherwise padding would leak into real positions. Left behind a flag,
# defaulting to the light MLP path.
#
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                 n_blocks=2, dropout=DROPOUT):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(input_dim, dropout=dropout) for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)      # mask MUST be passed -- these blocks attend
        return self.proj(self.norm(x)), mask


# Flip to True to use the heavier attention-based predictor instead of the MLP.
USE_TRANSFORMER_PREDICTOR = False


def build_predictor():
    if USE_TRANSFORMER_PREDICTOR:
        return TransformerPredictor()
    return Predictor()


# ---------------------------------------------------------------------------
# Sanity check: real encoder -> predictor on one real batch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    loader = make_dataloader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
    embedder = TokenEmbedding()
    encoder = RoPEEncoder()
    predictor = build_predictor()

    embedder.eval()
    encoder.eval()
    predictor.eval()

    tokens, mask, _ = next(iter(loader))   # contact maps unused by the predictor
    B, L = tokens.shape

    with torch.no_grad():
        embedded, mask = embedder(tokens, mask)
        encoded, mask = encoder(embedded, mask)
        predicted, mask_out = predictor(encoded, mask)

    print(f"batch shape:      {tuple(tokens.shape)}")
    print(f"encoded shape:    {tuple(encoded.shape)}")
    print(f"predicted shape:  {tuple(predicted.shape)}")

    assert predicted.shape == (B, L, OUTPUT_DIM), predicted.shape
    assert mask_out is mask, "mask was not passed straight through"
    assert not torch.isnan(predicted).any(), "predictor produced NaNs"
    assert not torch.isinf(predicted).any(), "predictor produced infs"

    print(f"sanity check passed: output (B, L, OUTPUT_DIM={OUTPUT_DIM}), no NaNs/infs.")
