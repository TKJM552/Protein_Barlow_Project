"""Contact-map encoder -- the JEPA "target" view.

This branch sees ONLY the contact map (never amino-acid identity) and produces a
per-residue representation that the predictor is trained to match via Barlow
Twins. It REUSES the pre-LN RoPE transformer block from seq_encoder.py; the single
difference from the sequence encoder is the attention mask -- here attention is
restricted to actual CONTACTS (plus self), not merely to non-padding.

forward signature is (padded_maps, mask) only -- there is deliberately no
seq_ints argument, so the encoder is blind to amino-acid identity.
"""

import torch
import torch.nn as nn

# Reuse the block and the shared attention hyperparameters from the sequence encoder.
from seq_encoder import (
    TransformerBlock,
    make_dataloader,
    BATCH_SIZE,
    MLP_HIDDEN,
    DROPOUT,
)
from config import DATA_DIR

# ---------------------------------------------------------------------------
# Constants (named, easy to change)
# ---------------------------------------------------------------------------
SEED_DIM = 2          # per-residue seed features: local degree + long-range degree
MODEL_DIM = 512
OUTPUT_DIM = 512      # MUST match the predictor's OUTPUT_DIM (Barlow Twins compares
                      # the two representations element-for-element).
N_BLOCKS = 4          # deliberately lighter than the 6-block sequence encoder
LOCAL_WINDOW = 6      # |i - j| <= LOCAL_WINDOW counts as short-range
# N_HEADS, HEAD_DIM, MLP_HIDDEN, DROPOUT, ROPE_BASE all live inside the reused block.


class ContactMapEncoder(nn.Module):
    def __init__(self, seed_dim=SEED_DIM, model_dim=MODEL_DIM, output_dim=OUTPUT_DIM,
                 n_blocks=N_BLOCKS, local_window=LOCAL_WINDOW, dropout=DROPOUT):
        super().__init__()
        self.local_window = local_window

        # Width-creating step, analogous to the sequence branch's embedding lookup.
        # NOTE: this is a plain Linear, so two residues with equal seed features map
        # to the EQUAL vector here on purpose -- the blocks differentiate residues by
        # WHO they contact (via the contact-restricted attention), not by the seed.
        self.seed_proj = nn.Linear(seed_dim, model_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, MLP_HIDDEN, dropout) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # Match the predictor's OUTPUT_DIM; identity when already equal.
        self.out_proj = (
            nn.Linear(model_dim, output_dim) if output_dim != model_dim else nn.Identity()
        )

    # ------------------------------------------------------------------
    # Seed features (B, L, SEED_DIM) -- computed from the MAP ONLY.
    # ------------------------------------------------------------------
    def _seed_features(self, padded_maps, mask):
        B, L, _ = padded_maps.shape
        device = padded_maps.device

        # Never count padding COLUMNS: zero them before summing over keys j.
        key_real = mask.view(B, 1, L).float()          # (B, 1, L), 1 = real key
        masked_map = padded_maps * key_real            # (B, L, L)

        # Distance bands over |i - j| (same for the whole batch).
        idx = torch.arange(L, device=device)
        dist = (idx.view(L, 1) - idx.view(1, L)).abs()  # (L, L)
        local_band = ((dist > 0) & (dist <= self.local_window)).float()  # exclude diagonal
        long_band = (dist > self.local_window).float()

        local_deg = (masked_map * local_band).sum(dim=-1)   # (B, L)
        long_deg = (masked_map * long_band).sum(dim=-1)     # (B, L)

        seeds = torch.stack((local_deg, long_deg), dim=-1)  # (B, L, 2)

        # log1p tames the scale of raw contact counts (a hub residue can have many
        # more contacts than a typical one); log1p(0) = 0 keeps zeros at zero.
        seeds = torch.log1p(seeds)

        # Zero the seed at padding ROWS too (belt-and-braces; those rows are 0 anyway).
        seeds = seeds * mask.unsqueeze(-1).float()
        return seeds

    # ------------------------------------------------------------------
    # Contact-restricted additive attention mask (B, 1, L, L).
    # ------------------------------------------------------------------
    def _contact_attn_mask(self, padded_maps, mask):
        B, L, _ = padded_maps.shape
        device = padded_maps.device

        # Allow query i -> key j only where (i,j) is a contact AND key j is real.
        allowed = (padded_maps > 0.5) & mask.view(B, 1, L)          # (B, L, L)

        # ALL-MIN-ROW GUARD: always allow the diagonal (i attends to itself) so every
        # query row has >= 1 valid key and softmax never sees an all-min row (-> NaN).
        # A contactless real residue therefore attends only to itself.
        eye = torch.eye(L, dtype=torch.bool, device=device).view(1, L, L)
        allowed = allowed | eye

        neg = torch.finfo(padded_maps.dtype).min
        add_mask = torch.zeros(B, L, L, dtype=padded_maps.dtype, device=device)
        add_mask = add_mask.masked_fill(~allowed, neg)              # 0 allowed, neg disallowed
        return add_mask.unsqueeze(1)                               # (B, 1, L, L)

    def forward(self, padded_maps, mask):
        # 1. Input assertions (do NOT recompute the map or the mask).
        assert padded_maps.dim() == 3 and padded_maps.dtype == torch.float32, \
            "padded_maps must be (B, L, L) float32"
        B, L, L2 = padded_maps.shape
        assert L == L2, "contact map must be square"
        assert mask.shape == (B, L) and mask.dtype == torch.bool, \
            "mask must be (B, L) bool matching padded_maps"
        # Cheap check that the diagonal is 1 at real positions (don't rebuild it).
        diag = torch.diagonal(padded_maps, dim1=-2, dim2=-1)        # (B, L)
        assert torch.all(diag[mask] == 1), "diagonal must be 1 at real positions"

        # 2-3. Seed features from the map only -> project to model width.
        seeds = self._seed_features(padded_maps, mask)             # (B, L, SEED_DIM)
        x = self.seed_proj(seeds)                                  # (B, L, MODEL_DIM)

        # 4. Contact-restricted attention mask.
        attn_mask = self._contact_attn_mask(padded_maps, mask)     # (B, 1, L, L)

        # 6. Reused pre-LN RoPE blocks, contact-restricted attention in each.
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        # 7. Match the predictor's OUTPUT_DIM.
        encoded = self.out_proj(x)
        return encoded, mask


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    loader = make_dataloader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
    encoder = ContactMapEncoder()
    encoder.eval()

    _, mask, padded_maps = next(iter(loader))   # NOTE: seq_ints deliberately dropped
    B, L = mask.shape

    # --- (a) Real batch -> output shape, no NaNs/infs ---------------------
    with torch.no_grad():
        encoded, mask_out = encoder(padded_maps, mask)

    print(f"padded_maps shape: {tuple(padded_maps.shape)}")
    print(f"encoded shape:     {tuple(encoded.shape)}")
    # How sparse is the sparsest map in this batch? (exercises the all-min-row guard)
    real_counts = mask.sum(dim=1)
    contacts = (padded_maps > 0.5).sum(dim=(1, 2)) - real_counts   # minus the diagonal
    print(f"sparsest map: {int(contacts.min())} off-diagonal contacts "
          f"(a residue with 0 falls back to self-attention only)")

    assert encoded.shape == (B, L, OUTPUT_DIM), encoded.shape
    assert mask_out is mask, "mask was not passed straight through"
    assert not torch.isnan(encoded).any(), "encoder produced NaNs"
    assert not torch.isinf(encoded).any(), "encoder produced infs"
    print("(a) shape ok, no NaNs/infs, mask passed through.")

    # --- (b) Blindness: the encoder cannot read amino-acid identity -------
    # Its forward signature is (padded_maps, mask) only -- there is no seq_ints
    # parameter, so it is structurally impossible for it to see the sequence.
    import inspect
    params = list(inspect.signature(encoder.forward).parameters)
    assert params == ["padded_maps", "mask"], f"unexpected forward signature: {params}"
    print(f"(b) blindness ok: forward takes {tuple(params)} -- no amino-acid input.")

    # --- (c) Padding check: corrupting the map at padding positions must not
    #         change outputs at REAL positions (padding keys/rows are masked out).
    real_pair = mask.view(B, L, 1) & mask.view(B, 1, L)            # True where both real
    rand_map = (torch.rand(B, L, L) > 0.5).float()
    # Keep the real x real block (incl. its 1-diagonal) intact; scramble the rest.
    corrupted = torch.where(real_pair, padded_maps, rand_map)
    with torch.no_grad():
        corrupt_encoded, _ = encoder(corrupted, mask)

    max_diff = (encoded[mask] - corrupt_encoded[mask]).abs().max().item()
    assert torch.allclose(encoded[mask], corrupt_encoded[mask], atol=1e-5), \
        f"padding leaked into real residues (max diff {max_diff:.2e})"
    print(f"(c) no leak: real positions unchanged when padding scrambled "
          f"(max diff {max_diff:.2e}).")

    print("sanity check passed: map encoder shape ok, blind to sequence, no padding leak.")
