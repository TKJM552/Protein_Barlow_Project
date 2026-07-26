#Embed inputs, then run through a RoPE self-attention encoder.

import os
import glob
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# BATCH_SIZE is re-exported here (rather than defined here) so the many existing
# `from seq_encoder import BATCH_SIZE` imports keep working while the value
# itself stays overridable via $BATCH_SIZE / --batch-size.
from config import BATCH_SIZE, DATA_DIR, NUM_WORKERS

VOCAB_SIZE = 21   # 0 = pad, 1-20 = aa's
PAD_IDX = 0
EMBED_DIM = 512

# --- encoder hyperparameters ---
N_HEADS = 8
HEAD_DIM = EMBED_DIM // N_HEADS   # 64
N_BLOCKS = 6
MLP_HIDDEN = 4 * EMBED_DIM        # 2048
DROPOUT = 0.1
ROPE_BASE = 10000


class ProteinSequenceDataset(Dataset):
    # .npz file -> (seq_ints 1-D long, contact_map (L,L) float32).
    # Skip files with zero-length sequences or a map that doesn't match seq length.

    def __init__(self, data_dir):
        self.paths = []
        for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            with np.load(path) as data:
                n = data["seq_ints"].shape[0]
                if n == 0:
                    continue  # get rid of empty sequences
                # seq_ints (length L_i) and contact_map (L_i, L_i) must agree.
                cmap_shape = tuple(data["contact_map"].shape)
                if cmap_shape != (n, n):
                    warnings.warn(
                        f"skipping {os.path.basename(path)}: seq length {n} does not "
                        f"match contact_map shape {cmap_shape}"
                    )
                    continue
                self.paths.append(path)

        if not self.paths:
            raise ValueError(f"No usable .npz files found in {data_dir!r}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as data:
            seq_ints = data["seq_ints"]        # int64 array, values in 1..20
            contact_map = data["contact_map"]  # int8 (L_i, L_i), symmetric, 1-diagonal
        seq = torch.as_tensor(seq_ints, dtype=torch.long)
        cmap = torch.as_tensor(contact_map, dtype=torch.float32)
        return seq, cmap


def collate_pad(batch):
    """Pad a batch of (seq_ints, contact_map) pairs to the longest sequence L in
    the batch so multiple proteins can be processed together.

    Returns:
        padded_ints: LongTensor (B, L), padding positions filled with PAD_IDX.
        mask:        BoolTensor (B, L). Convention: True  = real residue,
                                                     False = padding.
        padded_maps: FloatTensor (B, L, L). Each protein's (L_i, L_i) contact
                     map sits in the TOP-LEFT corner of an all-zeros (L, L)
                     block; padded rows/cols stay 0 (padding residues have no
                     contacts).
    """
    seqs = [seq for seq, _ in batch]
    maps = [cmap for _, cmap in batch]
    lengths = [seq.size(0) for seq in seqs]
    max_len = max(lengths)
    batch_size = len(batch)

    padded_ints = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    padded_maps = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)

    for i, length in enumerate(lengths):
        padded_ints[i, :length] = seqs[i]
        mask[i, :length] = True                      # mark real residues
        padded_maps[i, :length, :length] = maps[i]   # top-left corner

    return padded_ints, mask, padded_maps


def make_dataloader(data_dir=DATA_DIR, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=False):
    dataset = ProteinSequenceDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pad,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Workers each re-scan and re-open the .npz files on startup, which is the
        # expensive part; keep them alive across epochs instead of paying it 50x.
        persistent_workers=num_workers > 0,
    )



class TokenEmbedding(nn.Module):
    #Maps aa integer tokens to vectors.

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, pad_idx=PAD_IDX):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

    def forward(self, tokens, mask):
        """tokens: (B, L) longs -> returns ((B, L, EMBED_DIM) floats, mask).

        The mask is passed straight through so downstream stages can reuse it.
        """
        return self.embed(tokens), mask


# ---------------------------------------------------------------------------
# 5. RoPE utility (rotate-half convention)
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    #Rotary positional embedding, applied to Q and K per head over HEAD_DIM.

    #cos/sin tables are cached as float32 BUFFERS (not parameters): they move with
    #.to(device), are not learned, and are rebuilt on demand if a batch is longer
    #than the current cache.
 

    def __init__(self, head_dim=HEAD_DIM, base=ROPE_BASE):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base

        # inv_freq[i] = 1 / base**(2i/head_dim), for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cos/sin tables, built lazily / rebuilt when L grows.
        self._cached_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_tables(self, seq_len, device):
        # positions 0,1,2,...,L-1 along the sequence
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))          # (L, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (L, head_dim)
        self.cos_cached = emb.cos()                               # (L, head_dim), float32
        self.sin_cached = emb.sin()
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x):
        # split the last dim in half and rotate: (x1, x2) -> (-x2, x1)
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        """q, k: (B, N_HEADS, L, HEAD_DIM). Returns rotated (q, k) in input dtype.

        Rotation is done in float32 for mixed-precision safety, then cast back.
        """
        seq_len = q.shape[-2]
        if seq_len > self._cached_len or self.cos_cached.device != q.device:
            self._build_tables(seq_len, q.device)

        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, self.head_dim)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, self.head_dim)

        in_dtype = q.dtype
        q32, k32 = q.float(), k.float()
        q_rot = q32 * cos + self._rotate_half(q32) * sin
        k_rot = k32 * cos + self._rotate_half(k32) * sin
        return q_rot.to(in_dtype), k_rot.to(in_dtype)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x, B, L):
        # (B, L, E) -> (B, N_HEADS, L, HEAD_DIM)
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask):
        """x: (B, L, E). mask is one of two forms:
          * (B, L) bool  -> key-padding mask, True = real key (sequence encoder).
          * (B, 1, L, L) float -> additive per-(i,j) mask, 0 = allowed, large
            negative = disallowed (contact-map encoder). Broadcasts over heads.
        """
        B, L, _ = x.shape

        q = self._split_heads(self.q_proj(x), B, L)
        k = self._split_heads(self.k_proj(x), B, L)
        v = self._split_heads(self.v_proj(x), B, L)   # V is NOT rotated

        # RoPE into Q and K after projection, before the dot product.
        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, L, L)

        # Masking (the ONE spot that differs between the two encoders). Masked
        # positions get a large finite negative (finfo.min, NOT -inf, to avoid
        # NaNs from softmax on an all-masked row). Bidirectional -> no causal mask.
        if mask.dtype == torch.bool:
            key_mask = mask.view(B, 1, 1, L)                         # True = real key
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask                                   # (B, 1, L, L), broadcasts over heads

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                                  # (B, H, L, HEAD_DIM)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)        # (B, L, E)
        out = self.out_proj(out)
        return self.out_dropout(out)


# ---------------------------------------------------------------------------
# 7. Transformer block (pre-LN)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, mlp_hidden=MLP_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)   # pre-LN attention residual
        x = x + self.mlp(self.norm2(x))          # pre-LN MLP residual
        return x


# ---------------------------------------------------------------------------
# 8. Encoder stack
# ---------------------------------------------------------------------------
class RoPEEncoder(nn.Module):
    """Stack of N_BLOCKS pre-LN RoPE-attention blocks + a final LayerNorm.

    Consumes (embedded (B,L,E), mask (B,L)) and returns (encoded (B,L,E), mask),
    passing the same mask straight through.
    """

    def __init__(self, embed_dim=EMBED_DIM, n_blocks=N_BLOCKS,
                 mlp_hidden=MLP_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, mlp_hidden, dropout) for _ in range(n_blocks)
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- Deep-stack residual init (easy to remove) ---------------------
        # Scale the two residual-path output projections (attention out-proj and
        # the MLP's second Linear) by 1/sqrt(2*N_BLOCKS) so the residual stream
        # doesn't blow up as depth grows. Remove this loop to fall back to
        # PyTorch's default init everywhere.
        scale = 1.0 / math.sqrt(2 * n_blocks)
        for block in self.blocks:
            block.attn.out_proj.weight.data.mul_(scale)
            block.mlp[3].weight.data.mul_(scale)   # second Linear in the MLP
        # -------------------------------------------------------------------

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return x, mask


# 9. Sanity check
if __name__ == "__main__":
    # DATA_DIR comes from config (env-overridable); nothing is hardcoded here.

    # Deterministic + dropout OFF: eval() makes the two runs in the masking-leak
    # test comparable. With dropout active, each forward pass samples a different
    # dropout mask, so even identical inputs would produce different outputs and
    # the exact-match assertion would fail for reasons unrelated to leakage.
    torch.manual_seed(0)

    # =====================================================================
    # Dataset / collate check: sequences + padded contact maps
    # (shuffle=False so batch item i is dataset[i], letting us compare each
    #  protein against its original contact map.)
    # =====================================================================
    check_loader = make_dataloader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=False)
    dataset = check_loader.dataset
    padded_ints, m, padded_maps = next(iter(check_loader))
    Bc, Lc = padded_ints.shape

    print(f"padded_ints shape: {tuple(padded_ints.shape)}  dtype {padded_ints.dtype}")
    print(f"mask shape:        {tuple(m.shape)}  dtype {m.dtype}")
    print(f"padded_maps shape: {tuple(padded_maps.shape)}  dtype {padded_maps.dtype}")

    assert padded_ints.shape == (Bc, Lc) and padded_ints.dtype == torch.long
    assert m.shape == (Bc, Lc) and m.dtype == torch.bool
    assert padded_maps.shape == (Bc, Lc, Lc) and padded_maps.dtype == torch.float32

    # Consistency vs. originals (shuffle=False -> batch item i is dataset[i]).
    for i in range(min(3, Bc)):
        seq_i, cmap_i = dataset[i]
        Li = seq_i.size(0)
        assert int(m[i].sum()) == Li, "mask real-count != original seq length"
        assert torch.equal(padded_maps[i, :Li, :Li], cmap_i), "map block != original"

    # Symmetry, and padded rows/cols (mask False) are all zero.
    assert torch.equal(padded_maps, padded_maps.transpose(-1, -2)), "map not symmetric"
    pad_pos_c = ~m
    for i in range(Bc):
        if pad_pos_c[i].any():
            assert torch.all(padded_maps[i, pad_pos_c[i], :] == 0), "padded rows nonzero"
            assert torch.all(padded_maps[i, :, pad_pos_c[i]] == 0), "padded cols nonzero"
    print("dataset/collate check passed: shapes, consistency, symmetry, zero-padding.\n")

    # =====================================================================
    # Encoder checks (loader now yields a 3-tuple; the map is unused here)
    # =====================================================================
    loader = make_dataloader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
    embedder = TokenEmbedding()
    encoder = RoPEEncoder()
    embedder.eval()
    encoder.eval()

    tokens, mask, _ = next(iter(loader))   # contact maps unused by the encoder
    B, L = tokens.shape

    # --- Embedding-stage checks (as before) -------------------------------
    embeddings, mask_out = embedder(tokens, mask)
    assert embeddings.shape == (B, L, EMBED_DIM), embeddings.shape
    pad_positions = ~mask_out
    if pad_positions.any():
        assert torch.all(embeddings[pad_positions] == 0), "padding is not zero"
    assert torch.all(embeddings[mask_out].norm(dim=-1) > 0), "real residue is zero vec"

    # --- (a) Chain embedding -> encoder on a real batch -------------------
    with torch.no_grad():
        encoded, mask_passed = encoder(embeddings, mask_out)

    print(f"batch tokens shape:     {tuple(tokens.shape)}")
    print(f"encoded shape:          {tuple(encoded.shape)}")

    assert encoded.shape == (B, L, EMBED_DIM), encoded.shape
    assert mask_passed is mask_out, "mask was not passed straight through"
    assert not torch.isnan(encoded).any(), "encoder produced NaNs"
    assert not torch.isinf(encoded).any(), "encoder produced infs"
    print("(a) shape ok, no NaNs/infs, mask passed through.")

    # --- (b) Masking-leak test --------------------------------------------
    # Overwrite the integer IDs at PADDING positions with arbitrary non-zero
    # values, WITHOUT touching the mask, then re-embed and re-run. If padding
    # never leaks into real residues, outputs at REAL positions must be
    # unchanged. eval() (above) is what makes this an exact comparison: dropout
    # is deterministic (identity) in eval mode, so any difference can only come
    # from padding leaking through attention -- not from RNG.
    corrupted = tokens.clone()
    corrupted[pad_positions] = 7   # arbitrary non-zero (valid) amino-acid id
    with torch.no_grad():
        corrupt_emb, _ = embedder(corrupted, mask_out)
        corrupt_encoded, _ = encoder(corrupt_emb, mask_out)

    if pad_positions.any():
        max_diff = (encoded[mask_out] - corrupt_encoded[mask_out]).abs().max().item()
        assert torch.allclose(encoded[mask_out], corrupt_encoded[mask_out], atol=1e-5), \
            f"padding leaked into real residues (max diff {max_diff:.2e})"
        print(f"(b) no leak: real positions unchanged (max diff {max_diff:.2e}).")
    else:
        print("(b) skipped: uniform-length batch, no padding to corrupt.")

    print("sanity check passed: encoder shape correct, no NaNs, no padding leak.")
