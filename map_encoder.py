"""Contact-map encoder -- the structure branch.

This branch sees ONLY the contact map (never amino-acid identity) and produces a
per-residue representation (z_map) that Barlow Twins is asked to align with the
sequence branch's z_seq. Neither branch predicts the other.

It REUSES the pre-LN RoPE transformer block from seq_encoder.py; the single
difference from the sequence encoder is the attention mask -- here attention is
restricted to actual CONTACTS (plus self), not merely to non-padding.

forward signature is (padded_maps, mask) only -- there is deliberately no
seq_ints argument, so the encoder is blind to amino-acid identity.

SEEDING. Residue i enters the stack as its OWN ROW of the contact map -- the raw
binary vector of who it touches -- zero-padded to a fixed width and projected to
MODEL_DIM by one learned matrix. Nothing is summarised away first.

This replaces the previous 2-scalar seed (local degree, long-range degree), which
FINDINGS.md diagnosed as the run's ceiling: two non-negative scalars put every
residue on a 2-D cone, so a plain Linear mapped all L residues into a rank-2
subspace of the 512-d model space. The z_map vectors came out near-identical to each
other (mean pairwise cosine 0.42, effective rank 201) and matching them therefore
taught the sequence branch very little. A raw contact row is injective on contact
patterns: two residues get the same seed only if they contact the same partners.

The row is indexed by RELATIVE offset (j - i), not by absolute partner index j --
see SEED_MODE below for the measurement that forced that choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the block and the shared attention hyperparameters from the sequence encoder.
from seq_encoder import (
    TransformerBlock,
    make_dataloader,
    BATCH_SIZE,
    MLP_HIDDEN,
    DROPOUT,
)
from config import DATA_DIR, MAX_SEQ_LENGTH

# ---------------------------------------------------------------------------
# Constants (named, easy to change)
# ---------------------------------------------------------------------------
MAX_LEN = MAX_SEQ_LENGTH  # Longest chain this encoder can seed. MUST be >= the
                      # dataset's length cap -- so it now IS that cap, read from
                      # config.MAX_SEQ_LENGTH, which the RCSB query in
                      # get_files.py and the build-time re-check in
                      # get_inputs_outputs.check_length also read. Three copies of
                      # "1000" that had to be edited in lockstep are now one.
                      # This constant fixes a WEIGHT SHAPE -- changing it
                      # invalidates checkpoints.

# How the seed row is indexed. Both keep every contact; they differ in what a
# COLUMN of the projection means.
#
#   "relative" -- column k is the contact at OFFSET j - i. Width 2*MAX_LEN-1.
#   "absolute" -- column k is the contact at partner INDEX j. Width MAX_LEN.
#
# "relative" is the default because "absolute" leaks position badly. ~4 of every
# 10 contacts sit at |i-j| <= 2 and are present in 99.8% of those slots (CA-CA
# neighbours are ~3.8 A apart, always under the 8 A threshold). Under absolute
# indexing that always-on band lands in columns i-2..i+2 -- a pure function of i,
# identical in every protein. Measured on 120 chains, R^2 of predicting the seed
# from POSITION ALONE:
#
#                        seed    z_map at random init
#     absolute           0.573   0.620
#     relative           0.031   0.033
#     old 2-scalar seed  0.047   --
#
# 62% of an absolute-indexed z_map is reproducible from RoPE position alone, so
# the sequence branch could satisfy Barlow Twins without learning any structure.
# Relative indexing puts the always-on band at the SAME columns for every residue,
# where it is a shared constant rather than a position code, and costs only rank:
# within one 400-residue chain, mean pairwise seed cosine / matrix rank runs
# 0.891 / 2 (old 2-scalar), 0.024 / 396 (absolute), 0.611 / 217 (relative). Rank
# 2 -> 217 is the FINDINGS.md fix; absolute's extra spread is mostly position,
# not structure. Relative also trains its columns far more evenly -- see the
# gradient-coverage note in _seed_width.
SEED_MODE = "relative"

MODEL_DIM = 512
OUTPUT_DIM = 512      # MUST match the sequence encoder's EMBED_DIM: Barlow Twins
                      # compares the two branches dimension-for-dimension, and each
                      # expander is built for REP_DIM (barlow_twins.py).
N_BLOCKS = 4          # deliberately lighter than the 6-block sequence encoder
# N_HEADS, HEAD_DIM, MLP_HIDDEN, DROPOUT, ROPE_BASE all live inside the reused block.


def _seed_width(max_len, seed_mode):
    """Number of columns the seed projection needs.

    "absolute": one column per residue index          -> max_len
    "relative": one column per offset -(max_len-1) .. +(max_len-1) -> 2*max_len-1

    Gradient coverage differs sharply, and favours "relative". A column only gets
    gradient from a residue that actually has a partner there, so under "absolute"
    column k is fed only by chains longer than k: measured on the 4,966-chain
    dataset (median length 218) column 199 was fed by 2672 chains, column 499 by
    372, and column 899 by 16. Scaling to 150,000 chains (median 249) multiplies
    every one of those counts but not their ratio, so the argument is unchanged --
    the tail columns stay starved relative to the head. Under "relative" the
    columns that matter are the small
    offsets where contacts actually live, and offset d is fed by every one of the
    L-|d| residue pairs in every chain longer than |d| -- orders of magnitude more
    signal per column exactly where the contacts are.
    """
    return max_len if seed_mode == "absolute" else 2 * max_len - 1


class ContactMapEncoder(nn.Module):
    def __init__(self, max_len=MAX_LEN, model_dim=MODEL_DIM, output_dim=OUTPUT_DIM,
                 n_blocks=N_BLOCKS, dropout=DROPOUT, seed_mode=SEED_MODE):
        super().__init__()
        if seed_mode not in ("relative", "absolute"):
            raise ValueError(f"seed_mode must be 'relative' or 'absolute', got {seed_mode!r}")
        self.max_len = max_len
        self.seed_mode = seed_mode

        # Width-creating step, analogous to the sequence branch's embedding lookup.
        # A residue's seed is the SUM of the columns of its contact partners (plus
        # bias), so two residues share a seed only if they share a contact pattern
        # -- contrast the old 2-scalar seed, where equal degrees forced an equal
        # vector by construction.
        self.seed_proj = nn.Linear(_seed_width(max_len, seed_mode), model_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, MLP_HIDDEN, dropout) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # See the diagonal check in forward(): validated on the first batch only.
        self._diagonal_checked = False

        # Match the sequence branch's width; identity when already equal.
        self.out_proj = (
            nn.Linear(model_dim, output_dim) if output_dim != model_dim else nn.Identity()
        )

    # ------------------------------------------------------------------
    # Seed rows (B, L, L) -- lifted straight out of the MAP, nothing else.
    # ------------------------------------------------------------------
    def _seed_rows(self, padded_maps, mask):
        """Residue i's own contact row, zeroed outside the real x real block.

        Two things are masked, for two different reasons:
          * padding COLUMNS (keys) -- a real residue's seed must not depend on what
            sits at a padding index, or garbage there leaks into a real output.
            This is the load-bearing one now that the whole row is the seed.
          * padding ROWS -- belt-and-braces; collate_pad already leaves them 0.

        The map is symmetric, so "rows" and "columns" are the same vectors; rows
        are used because padded_maps[b, i, :] is the contiguous one.
        """
        real_pair = mask.unsqueeze(-1) & mask.unsqueeze(1)     # (B, L, L), True = both real
        return padded_maps * real_pair

    # ------------------------------------------------------------------
    # Absolute-indexed rows -> offset-indexed rows.
    # ------------------------------------------------------------------
    @staticmethod
    def _to_relative(rows):
        """(B, L, L) rows indexed by partner INDEX -> (B, L, 2L-1) indexed by OFFSET.

        out[b, i, k] = rows[b, i, i + k - (L-1)], so column k holds the contact at
        relative offset k - (L-1) and offset 0 (the self-contact) always lands in
        the middle column. Partners that fall off either end of the chain are 0.

        The index tensor is rebuilt per call rather than cached: it is two aranges,
        negligible next to the gather and the matmul that follow.
        """
        B, L, _ = rows.shape
        device = rows.device
        i = torch.arange(L, device=device).view(L, 1)
        k = torch.arange(2 * L - 1, device=device).view(1, 2 * L - 1)
        j = i + k - (L - 1)                                    # (L, 2L-1) partner index
        valid = (j >= 0) & (j < L)
        # clamp() keeps gather in bounds; `* valid` then discards those reads, which
        # is why the clamp cannot smuggle in a real contact from column 0 or L-1.
        j = j.clamp(0, L - 1).unsqueeze(0).expand(B, L, 2 * L - 1)
        return rows.gather(2, j) * valid

    # ------------------------------------------------------------------
    # Seed rows -> model width, via the fixed-width projection.
    # ------------------------------------------------------------------
    def _project_seed(self, rows):
        """(B, L, L) contact rows -> (B, L, MODEL_DIM).

        Semantically: pad the seed row with zeros out to the projection's full
        width, then apply seed_proj. Implemented by slicing the weight to the
        columns a length-L batch can actually reach, which is EXACTLY equal -- the
        padded entries are 0, so the omitted columns contribute nothing to the sum
        -- while materialising no full-width tensor. Sanity check (d) asserts the
        equality numerically, in both seed modes.

        Cost note: the matmul is ~8 GFLOP (absolute) or ~16 GFLOP (relative) at
        B=16, L=1000, i.e. at most one attention block's worth. In relative mode
        the gather also materialises a (B, L, 2L-1) tensor -- 128 MB fp32 at that
        worst case, less than the (B, H, L, L) score matrix one block already
        holds, and ~11 MB at the dataset's median length.
        """
        L = rows.shape[-1]
        if self.seed_mode == "absolute":
            # Column k is partner index k, so a length-L batch reaches columns 0..L-1.
            return F.linear(rows, self.seed_proj.weight[:, :L], self.seed_proj.bias)

        # Column k is offset k - (max_len-1). A length-L batch reaches offsets
        # -(L-1)..+(L-1), i.e. the 2L-1 columns centred on the offset-0 column.
        rel = self._to_relative(rows)                          # (B, L, 2L-1)
        weight = self.seed_proj.weight[:, self.max_len - L:self.max_len + L - 1]
        return F.linear(rel, weight, self.seed_proj.bias)

    # ------------------------------------------------------------------
    # Contact-restricted attention mask (B, 1, L, L) bool, True = attend.
    # ------------------------------------------------------------------
    def _contact_attn_mask(self, padded_maps, mask):
        """Bool, deliberately: this used to be a float additive mask built at
        padded_maps.dtype (fp32). Adding that to fp16 attention scores promoted
        the whole (B, H, L, L) score tensor to fp32 -- 502 MB instead of 251 MB at
        B=16, L=990, on every one of this encoder's blocks, with the AMP speedup
        thrown away. A bool mask is also 4x smaller than the fp32 one it replaces.
        """
        B, L, _ = padded_maps.shape
        device = padded_maps.device

        # Allow query i -> key j only where (i,j) is a contact AND key j is real.
        allowed = (padded_maps > 0.5) & mask.view(B, 1, L)          # (B, L, L)

        # ALL-MIN-ROW GUARD: always allow the diagonal (i attends to itself) so every
        # query row has >= 1 valid key and softmax never sees an all-min row (-> NaN).
        # A contactless real residue therefore attends only to itself.
        eye = torch.eye(L, dtype=torch.bool, device=device).view(1, L, L)
        allowed = allowed | eye

        return allowed.unsqueeze(1)                                 # (B, 1, L, L)

    def forward(self, padded_maps, mask):
        # 1. Input assertions (do NOT recompute the map or the mask).
        assert padded_maps.dim() == 3 and padded_maps.dtype == torch.float32, \
            "padded_maps must be (B, L, L) float32"
        B, L, L2 = padded_maps.shape
        assert L == L2, "contact map must be square"
        assert mask.shape == (B, L) and mask.dtype == torch.bool, \
            "mask must be (B, L) bool matching padded_maps"
        assert L <= self.max_len, (
            f"batch length {L} exceeds MAX_LEN={self.max_len}: the seed projection "
            f"has a fixed number of columns and there is none to read residue "
            f"{L - 1} from. Either keep the dataset's length cap at or below MAX_LEN "
            f"(get_inputs_outputs.build_dataset) or raise MAX_LEN and retrain -- the "
            f"weight shape, and so every checkpoint, depends on it."
        )
        # Diagonal must be 1 at real positions. Checked on the FIRST batch only:
        # unlike the assertions above (which read shapes and dtypes on the host),
        # this one reads tensor VALUES, which forces a device->host sync every
        # step and serialises exactly the H2D prefetch that pin_memory +
        # non_blocking are there to overlap. Malformed data is a property of the
        # dataset, not of a particular batch, so one check catches it.
        if not self._diagonal_checked:
            diag = torch.diagonal(padded_maps, dim1=-2, dim2=-1)    # (B, L)
            assert torch.all(diag[mask] == 1), "diagonal must be 1 at real positions"
            self._diagonal_checked = True

        # 2-3. Seed = each residue's raw contact row -> project to model width.
        rows = self._seed_rows(padded_maps, mask)                  # (B, L, L)
        x = self._project_seed(rows)                               # (B, L, MODEL_DIM)

        # 4. Contact-restricted attention mask.
        attn_mask = self._contact_attn_mask(padded_maps, mask)     # (B, 1, L, L)

        # 6. Reused pre-LN RoPE blocks, contact-restricted attention in each.
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)

        # 7. Match the sequence branch's width.
        encoded = self.out_proj(x)
        return encoded, mask


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    loader = make_dataloader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
    _, mask, padded_maps = next(iter(loader))   # NOTE: seq_ints deliberately dropped
    B, L = mask.shape

    print(f"padded_maps shape: {tuple(padded_maps.shape)}   (B={B}, L={L})")
    # How sparse is the sparsest map in this batch? (exercises the all-min-row guard)
    real_counts = mask.sum(dim=1)
    contacts = (padded_maps > 0.5).sum(dim=(1, 2)) - real_counts   # minus the diagonal
    print(f"sparsest map: {int(contacts.min())} off-diagonal contacts "
          f"(a residue with 0 falls back to self-attention only)")

    # =====================================================================
    # (a) / (c) / (d) / (e) run for BOTH seed modes -- the non-default path
    # is kept working so the two can be A/B'd on a real run.
    # =====================================================================
    def check_mode(mode):
        torch.manual_seed(0)
        encoder = ContactMapEncoder(seed_mode=mode)
        encoder.eval()
        width = encoder.seed_proj.in_features
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\n--- seed_mode={mode!r}  (seed_proj is {MODEL_DIM}x{width}, "
              f"encoder {n_params/1e6:.2f}M params) ---")

        # --- (a) Real batch -> output shape, no NaNs/infs ------------------
        with torch.no_grad():
            encoded, mask_out = encoder(padded_maps, mask)
        assert encoded.shape == (B, L, OUTPUT_DIM), encoded.shape
        assert mask_out is mask, "mask was not passed straight through"
        assert not torch.isnan(encoded).any(), "encoder produced NaNs"
        assert not torch.isinf(encoded).any(), "encoder produced infs"
        print(f"(a) shape {tuple(encoded.shape)} ok, no NaNs/infs, mask passed through.")

        # --- (c) Corrupting the map at PADDING positions must not change any
        #         REAL residue. This matters more than it used to: the whole row
        #         is the seed now, so without the column mask in _seed_rows, junk
        #         at a padding index would land straight in a real output.
        real_pair = mask.view(B, L, 1) & mask.view(B, 1, L)
        rand_map = (torch.rand(B, L, L) > 0.5).float()
        corrupted = torch.where(real_pair, padded_maps, rand_map)
        with torch.no_grad():
            corrupt_encoded, _ = encoder(corrupted, mask)
        d = (encoded[mask] - corrupt_encoded[mask]).abs().max().item()
        assert torch.allclose(encoded[mask], corrupt_encoded[mask], atol=1e-5), \
            f"padding leaked into real residues (max diff {d:.2e})"
        print(f"(c) no leak: real positions unchanged when padding scrambled "
              f"(max diff {d:.2e}).")

        # --- (d) The weight slice IS zero-padding to full width ------------
        # _project_seed slices seed_proj.weight rather than building a full-width
        # tensor. Prove the two agree by doing the padding explicitly.
        rows = encoder._seed_rows(padded_maps, mask)
        explicit = torch.zeros(B, L, width)
        if mode == "absolute":
            explicit[:, :, :L] = rows                       # right-pad with zeros
        else:
            # centre the reachable offsets on the offset-0 column
            explicit[:, :, MAX_LEN - L:MAX_LEN + L - 1] = encoder._to_relative(rows)
        with torch.no_grad():
            via_slice = encoder._project_seed(rows)
            via_pad = encoder.seed_proj(explicit)
        d = (via_slice - via_pad).abs().max().item()
        assert torch.allclose(via_slice, via_pad, atol=1e-5), \
            f"weight slice != explicit zero-pad (max diff {d:.2e})"
        print(f"(d) weight slice == explicit zero-pad to width {width} "
              f"(max diff {d:.2e}).")

        # --- (e) Batch-composition invariance ------------------------------
        # The seed width is a FIXED function of MAX_LEN, not of the batch's longest
        # chain, so a protein's representation does not depend on who it was
        # batched with. Add padding columns and check every real residue is
        # unchanged. This is why the pad target is a constant, not max(lengths).
        pad_extra = min(64, MAX_LEN - L)
        if pad_extra > 0:
            W = L + pad_extra
            wide_maps = torch.zeros(B, W, W)
            wide_maps[:, :L, :L] = padded_maps
            wide_mask = torch.zeros(B, W, dtype=torch.bool)
            wide_mask[:, :L] = mask
            with torch.no_grad():
                wide_encoded, _ = encoder(wide_maps, wide_mask)
            d = (encoded[mask] - wide_encoded[wide_mask]).abs().max().item()
            assert torch.allclose(encoded[mask], wide_encoded[wide_mask], atol=1e-5), \
                f"output depends on batch padding width (max diff {d:.2e})"
            print(f"(e) batch-composition invariant: +{pad_extra} padding columns "
                  f"change nothing at real positions (max diff {d:.2e}).")
        else:
            print(f"(e) skipped: batch already at MAX_LEN={MAX_LEN}, no room to widen.")
        return encoder, rows

    encoder, rows = check_mode(SEED_MODE)      # the default, used by (f)/(g) below
    check_mode("absolute" if SEED_MODE == "relative" else "relative")

    # --- (b) Blindness: the encoder cannot read amino-acid identity -------
    # Its forward signature is (padded_maps, mask) only -- there is no seq_ints
    # parameter, so it is structurally impossible for it to see the sequence.
    import inspect
    params = list(inspect.signature(encoder.forward).parameters)
    assert params == ["padded_maps", "mask"], f"unexpected forward signature: {params}"
    print(f"\n(b) blindness ok: forward takes {tuple(params)} -- no amino-acid input.")

    # =====================================================================
    # (f) Seed diversity -- the problem this seeding was built to fix.
    # (g) Position leakage -- the problem RELATIVE indexing was built to fix.
    # Both compare against the old 2-scalar seed, recomputed here so the
    # comparison stays runnable after that code path was deleted.
    # =====================================================================
    LOCAL_WINDOW = 6                                   # the old local/long split

    def old_seed_for(m):
        """The deleted 2-scalar seed: log1p(local degree), log1p(long-range degree)."""
        n = m.shape[-1]
        idx = torch.arange(n)
        dist = (idx.view(n, 1) - idx.view(1, n)).abs()
        return torch.log1p(torch.stack((
            (m * ((dist > 0) & (dist <= LOCAL_WINDOW)).float()).sum(-1),
            (m * (dist > LOCAL_WINDOW).float()).sum(-1),
        ), dim=-1))

    b = int(mask.sum(dim=1).argmax())
    Lb = int(mask[b].sum())
    m_b = padded_maps[b, :Lb, :Lb]

    def diversity(M):
        """Mean pairwise cosine between residues' seeds, and the matrix rank."""
        U = M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        C = U @ U.T
        off = ~torch.eye(M.shape[0], dtype=torch.bool)
        return C[off].mean().item(), int(torch.linalg.matrix_rank(M))

    print(f"\n(f) seed diversity, longest protein in the batch (L={Lb}). The old seed's "
          f"rank-2\n    collapse is what FINDINGS.md blamed for homogeneous z_map:")
    for name, M in (("old 2-scalar", old_seed_for(m_b)),
                    ("absolute row", m_b),
                    ("relative row", ContactMapEncoder._to_relative(m_b.unsqueeze(0))[0])):
        cos, rank = diversity(M)
        print(f"    {name:13s} width {M.shape[1]:4d}  mean pairwise cos {cos:.3f}  "
              f"rank {rank:4d}")
    print(f"    contact density: {m_b.sum(-1).mean():.1f} contacts/residue, of which "
          f"{(m_b * ((torch.arange(Lb).view(Lb,1) - torch.arange(Lb).view(1,Lb)).abs() <= 2)).sum(-1).mean():.1f} "
          f"at |i-j| <= 2")

    # --- (g) How much of the seed is predictable from POSITION alone? -----
    # Load a few same-length chains and ask: what fraction of the variance across
    # (protein, position) is explained by the position-conditional mean? A high
    # number means the sequence branch can match z_map from RoPE position
    # without learning any structure -- a shortcut that would inflate every
    # agreement metric in eval.py. This is the measurement behind SEED_MODE.
    import glob
    import os

    import numpy as np

    # NOTE: R^2 here is upward-biased by the chain count -- the position-conditional
    # mean is estimated from P proteins, so noise in it reads as explained variance
    # (~1/P). At P=80 that floor is ~0.01, enough to separate 0.03 from 0.6 but NOT
    # enough to compare these numbers against a run with a different P. Same
    # small-sample trap FINDINGS.md records for CKA.
    L0, N_PROT = 150, 80
    same_len = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.npz"))):
        with np.load(path) as d:
            n = d["seq_ints"].shape[0]
            if n < L0 or tuple(d["contact_map"].shape) != (n, n):
                continue
            same_len.append(torch.as_tensor(d["contact_map"][:L0, :L0],
                                            dtype=torch.float32))
        if len(same_len) == N_PROT:
            break
    maps_fixed = torch.stack(same_len)                     # (P, L0, L0)

    def r2_position(X):
        """X: (P, L, D) -> fraction of variance explained by position alone."""
        X = X.double()
        grand = X.reshape(-1, X.shape[-1]).mean(0)
        total = ((X - grand) ** 2).sum(-1).mean()
        return (((X.mean(0) - grand) ** 2).sum(-1).mean() / total).item()

    print(f"\n(g) R^2 of predicting the seed from POSITION ALONE "
          f"({len(same_len)} chains, first {L0} residues).\n"
          f"    HIGH is bad: it is matchable from RoPE position without learning structure.")
    for name, X in (("old 2-scalar", old_seed_for(maps_fixed)),
                    ("absolute row", maps_fixed),
                    ("relative row", ContactMapEncoder._to_relative(maps_fixed))):
        print(f"    {name:13s} R^2 {r2_position(X):.3f}")
    rel_r2 = r2_position(ContactMapEncoder._to_relative(maps_fixed))
    abs_r2 = r2_position(maps_fixed)
    assert rel_r2 < abs_r2, "relative indexing should leak LESS position than absolute"
    print(f"    -> relative leaks {abs_r2/max(rel_r2,1e-9):.0f}x less position than "
          f"absolute; SEED_MODE={SEED_MODE!r}")

    print("\nsanity check passed: shapes ok, blind to sequence, no padding leak, "
          "both seed modes work.")
