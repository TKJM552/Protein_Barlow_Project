"""Per-residue vectors -> contact map. The downstream model Barlow Twins feeds.

    (B, L, 512) residue vectors + mask  ->  (B, L, L) contact logits

The input is deliberately SOURCE-AGNOSTIC. Both arms of the experiment that
justifies this project produce (B, L, 512):

    arm A   frozen z_seq from a pretrained sequence_encoder
    arm B   nn.Embedding(21, 512) on the raw amino acids, trained from scratch

so the two runs share this file byte for byte and the only difference is what
feeds it. **The delta between those arms is the result** -- nothing else in this
repo establishes that pretraining is worth anything downstream, and a model that
can only be run in arm A cannot answer the question.

WHY THIS IS THE FRAGILE PART, not the robust one
------------------------------------------------
The L^2 output is where this gets hard, in three ways worth knowing before the
first run:

1. CLASS IMBALANCE. Contacts are ~2-5% of pairs. Unweighted BCE reaches ~96%
   accuracy by predicting "no contact" everywhere, and its loss curve looks
   healthy the whole time. `contact_loss` takes a pos_weight for this, and
   `pos_weight_from_maps` measures it off your own data rather than guessing.

2. THE |i-j| SHORTCUT. Sequence-adjacent residues are nearly always in contact
   (~4 of every 10 contacts sit at |i-j| <= 2 -- consecutive CA atoms are ~3.8 A
   apart). FINDINGS records that this trivial prior alone scores AUC 0.758,
   *beating* the pretrained encoder's 0.636. So a pair model given relative
   position can score well having never read the sequence at all.

   `use_relpos` is therefore a flag, not a fixture. Leave it ON (the prior is
   real information and withholding it just makes the model relearn it), but
   judge on **long-range** contacts, |i-j| > 12, where it carries nothing --
   and run `--use-relpos false` once as the control that says how much of your
   number came from the model versus from the offset embedding.

3. MEMORY IS L^2. One (B=4, L=990, C=64) pair tensor is 1 GB, and each axial
   block holds several. Train on random square CROPS (see crop_pair below);
   evaluate on whole proteins under no_grad. The defaults here are deliberately
   small -- 2.1M parameters against the encoder's 43.1M -- because this repo's
   measured failure is memorisation, and a fat pair track would reach it faster.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq_encoder import EMBED_DIM, TransformerBlock

# --- dimensions ------------------------------------------------------------
IN_DIM = EMBED_DIM         # 512. What both arms hand over; NOT independent of
                           # seq_encoder -- z_seq is this wide by construction.
SEQ_DIM = 256              # 1D track width
N_SEQ_BLOCKS = 2           # shallow ON PURPOSE: in arm A the input is already
                           # contextualised by a 6-block encoder, so depth here
                           # mostly buys capacity to memorise. Arm B gets the
                           # same 2 blocks -- an unfair advantage to arm A, and
                           # the fair-comparison cost of a shared architecture.
PAIR_DIM = 64              # every activation is (B, L, L, PAIR_DIM); this is
                           # the number that decides whether a batch fits
N_PAIR_BLOCKS = 4
N_HEADS = 4
MLP_RATIO = 4
DROPOUT = 0.1

RELPOS_CLIP = 32           # |i-j| beyond this shares one embedding. Past ~32 the
                           # exact offset stops being informative on its own, and
                           # a wider table is a wider shortcut.

CROP = 192                 # residues per side during training. Peak pair memory
                           # goes as CROP^2, so this is the OOM knob.


# ---------------------------------------------------------------------------
# Pair featurisation -- where (B, L, D) becomes (B, L, L, C)
# ---------------------------------------------------------------------------
class PairFeaturizer(nn.Module):
    """h_i, h_j -> p_ij, by outer SUM rather than outer product.

    p_ij = W_a h_i + W_b h_j (+ relpos(i-j))

    The outer sum is used instead of an outer product because it is O(L^2 * C)
    rather than O(L^2 * D^2) and because the axial blocks that follow can build
    multiplicative interactions themselves. Two separate projections, not one
    shared: p_ij must be able to differ from p_ji before symmetrisation, or the
    pair track starts symmetric and can never represent a directional feature.
    """

    def __init__(self, seq_dim=SEQ_DIM, pair_dim=PAIR_DIM, use_relpos=True,
                 relpos_clip=RELPOS_CLIP):
        super().__init__()
        self.proj_a = nn.Linear(seq_dim, pair_dim)
        self.proj_b = nn.Linear(seq_dim, pair_dim)
        self.use_relpos = use_relpos
        self.relpos_clip = relpos_clip
        if use_relpos:
            # 2 * clip + 1 buckets: every offset from -clip to +clip, ends saturating.
            self.relpos = nn.Embedding(2 * relpos_clip + 1, pair_dim)

    def forward(self, h):
        B, L, _ = h.shape
        p = self.proj_a(h).unsqueeze(2) + self.proj_b(h).unsqueeze(1)   # (B,L,L,C)
        if self.use_relpos:
            idx = torch.arange(L, device=h.device)
            off = (idx[:, None] - idx[None, :]).clamp(-self.relpos_clip,
                                                      self.relpos_clip)
            p = p + self.relpos(off + self.relpos_clip)
        return p


# ---------------------------------------------------------------------------
# Axial attention -- the pair track's transformer
# ---------------------------------------------------------------------------
class AxialAttention(nn.Module):
    """Multi-head self-attention over ONE axis of the pair tensor.

    Operates on (N, L, C) where N folds together the batch and the axis being
    held fixed. Full attention over an (L, L) grid would be O(L^4); attending
    along rows and then columns is O(L^3) and is what the Evoformer's pair track
    does for the same reason.
    """

    def __init__(self, dim=PAIR_DIM, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"pair_dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, key_mask):
        N, L, C = x.shape
        qkv = (self.qkv(x)
               .view(N, L, 3, self.n_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))                    # (3, N, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # A fully-padded key row would make softmax divide by zero and return
        # NaN. It cannot happen here -- key_mask is the protein's residue mask,
        # so it holds at least one True for any protein with any residue -- but
        # the guard costs nothing and the failure would be silent NaN weights.
        safe = key_mask.clone()
        safe[~safe.any(dim=1)] = True

        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=safe[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(o.transpose(1, 2).reshape(N, L, C))


class AxialBlock(nn.Module):
    """Pre-LN row attention, then column attention, then FFN. Each residual."""

    def __init__(self, pair_dim=PAIR_DIM, n_heads=N_HEADS, mlp_ratio=MLP_RATIO,
                 dropout=DROPOUT):
        super().__init__()
        self.norm_row = nn.LayerNorm(pair_dim)
        self.row_attn = AxialAttention(pair_dim, n_heads, dropout)
        self.norm_col = nn.LayerNorm(pair_dim)
        self.col_attn = AxialAttention(pair_dim, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(pair_dim)
        self.ffn = nn.Sequential(
            nn.Linear(pair_dim, mlp_ratio * pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * pair_dim, pair_dim),
            nn.Dropout(dropout),
        )

    def forward(self, p, mask):
        B, L, _, C = p.shape
        # Same key mask for both axes: it is the residue mask either way, just
        # expanded along whichever index is being held fixed.
        km = mask[:, None, :].expand(B, L, L).reshape(B * L, L)

        h = self.norm_row(p).reshape(B * L, L, C)
        p = p + self.row_attn(h, km).view(B, L, L, C)

        h = self.norm_col(p).transpose(1, 2).reshape(B * L, L, C)
        p = p + self.col_attn(h, km).view(B, L, L, C).transpose(1, 2)

        return p + self.ffn(self.norm_ffn(p))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class ContactPredictor(nn.Module):
    """(B, L, IN_DIM) residue vectors + (B, L) mask -> (B, L, L) contact logits.

    Returns LOGITS, not probabilities: the loss is BCEWithLogits, which is the
    numerically stable form. Call .sigmoid() for probabilities at eval time.

    Output is explicitly symmetrised. A contact map is symmetric by definition,
    so letting the model produce an asymmetric one wastes half its capacity
    learning a constraint that can be imposed for free -- and makes P@L/5, which
    reads the upper triangle, depend on which triangle you read.
    """

    def __init__(self, in_dim=IN_DIM, seq_dim=SEQ_DIM, pair_dim=PAIR_DIM,
                 n_seq_blocks=N_SEQ_BLOCKS, n_pair_blocks=N_PAIR_BLOCKS,
                 n_heads=N_HEADS, dropout=DROPOUT, use_relpos=True):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, seq_dim)
        # Reused from seq_encoder rather than reimplemented: same pre-LN block,
        # same RoPE attention, same masking semantics as the encoder upstream.
        self.seq_blocks = nn.ModuleList(
            TransformerBlock(seq_dim, MLP_RATIO * seq_dim, dropout)
            for _ in range(n_seq_blocks)
        )
        self.seq_norm = nn.LayerNorm(seq_dim)

        self.pair_init = PairFeaturizer(seq_dim, pair_dim, use_relpos)
        self.pair_blocks = nn.ModuleList(
            AxialBlock(pair_dim, n_heads, MLP_RATIO, dropout)
            for _ in range(n_pair_blocks)
        )
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.out = nn.Linear(pair_dim, 1)

        # Same deep-stack residual scaling as RoPEEncoder: shrink the residual
        # branches' output projections so the stream does not grow with depth.
        scale = 1.0 / math.sqrt(2 * max(n_seq_blocks + n_pair_blocks, 1))
        for b in self.seq_blocks:
            b.attn.out_proj.weight.data.mul_(scale)
            b.mlp[3].weight.data.mul_(scale)
        for b in self.pair_blocks:
            b.row_attn.out_proj.weight.data.mul_(scale)
            b.col_attn.out_proj.weight.data.mul_(scale)
            b.ffn[3].weight.data.mul_(scale)

        # Start predicting "no contact" everywhere. Contacts are a few percent of
        # pairs, so a default-init head starts near p=0.5 and the first hundred
        # steps are spent walking the bias down instead of learning anything.
        # The weight is small but NOT zero: a zero weight makes the output a
        # constant, which trains fine but is indistinguishable from a model that
        # ignores its input -- check (d) below exists to catch exactly that, and
        # it cannot if the healthy state also fails it.
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.constant_(self.out.bias, -3.0)   # sigmoid(-3) ~ 0.047

    def forward(self, x, mask):
        h = self.in_proj(x)
        for block in self.seq_blocks:
            h = block(h, mask)
        h = self.seq_norm(h)

        p = self.pair_init(h)
        for block in self.pair_blocks:
            p = block(p, mask)

        logits = self.out(self.pair_norm(p)).squeeze(-1)      # (B, L, L)
        logits = 0.5 * (logits + logits.transpose(1, 2))

        # Zero the padded block so nothing downstream reads garbage there. The
        # loss masks it anyway; this makes the returned tensor safe to look at.
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return logits.masked_fill(~pair_mask, 0.0)


# ---------------------------------------------------------------------------
# Loss and the two things it has to be told
# ---------------------------------------------------------------------------
def pair_mask_for(mask, min_sep=0):
    """(B,L) residue mask -> (B,L,L) pair mask, optionally dropping |i-j| < min_sep.

    min_sep=12 restricts to the long-range contacts P@L/5 is scored on. That is a
    METRIC restriction, not a training one -- see contact_loss. Cutting the short
    band out of the loss would leave the model unable to predict it, and a contact
    map with a hole down its diagonal is not a contact map.
    """
    pm = mask[:, :, None] & mask[:, None, :]
    if min_sep > 0:
        L = mask.shape[1]
        idx = torch.arange(L, device=mask.device)
        pm = pm & ((idx[:, None] - idx[None, :]).abs() >= min_sep)
    return pm


def pos_weight_from_maps(maps, mask, min_sep=0):
    """negatives / positives over the real pairs. Measured, not guessed.

    Feed this to contact_loss. It varies with min_sep by roughly an order of
    magnitude -- long-range contacts are far rarer than all contacts -- so a
    weight computed at min_sep=0 is badly wrong for a long-range-only loss.
    """
    pm = pair_mask_for(mask, min_sep)
    pos = (maps * pm).sum()
    total = pm.sum()
    return ((total - pos) / pos.clamp_min(1.0)).clamp_min(1.0)


def long_weight_from_maps(maps, mask, min_sep=12):
    """How much to upweight long-range pairs so both bands teach equally.

    The problem this solves, measured over 300 proteins of the committed dataset:

        contacts at |i-j| < 12        72.3%   <- nearly free; consecutive CA atoms
                                                are ~3.8 A apart against an 8 A
                                                threshold, so proximity in the
                                                chain almost forces contact
        non-contacts per contact      45.1    <- what pos_weight corrects
        short positives / long        2.61    <- what THIS corrects

    So with a flat loss, **72% of the gradient on positives goes to the band a
    model can get from |i-j| alone**, and the long-range contacts that actually
    require reading the sequence are the minority of the signal.

    The wrong fix is to drop the short band from the loss; the model then cannot
    predict it and the output map is unusable. The right fix is to rebalance, and
    the natural target is that each band contributes equal POSITIVE mass:

        weight = (short-range positives) / (long-range positives)

    Measured off the batch, like pos_weight, rather than picked. Returns 1.0 when
    the two bands already balance, so passing it is never worse than not.
    """
    short = pair_mask_for(mask, 0) & ~pair_mask_for(mask, min_sep)
    long_ = pair_mask_for(mask, min_sep)
    short_pos = (maps * short).sum()
    long_pos = (maps * long_).sum()
    return (short_pos / long_pos.clamp_min(1.0)).clamp_min(1.0)


def separation_weight(mask, min_sep=12, long_weight=1.0):
    """(B,L,L) per-pair multiplier: 1.0 inside the short band, long_weight outside.

    Deliberately a two-band step rather than a smooth function of |i-j|: the
    physics is a step. Below ~4 residues contact is near-certain, and past ~12 the
    offset carries essentially nothing. A smooth ramp would imply a gradation the
    data does not have, and adds a shape to tune.
    """
    if long_weight == 1.0:
        return None
    w = torch.ones(mask.shape[0], mask.shape[1], mask.shape[1], device=mask.device)
    return w.masked_fill(pair_mask_for(mask, min_sep), float(long_weight))


def contact_loss(logits, target, mask, pos_weight=None, sep_weight=None, min_sep=0):
    """Masked, class-weighted BCE over the real pairs.

    Two independent weights, doing two different jobs:

      pos_weight  a scalar. Contacts vs non-contacts -- stops the model scoring
                  ~97% by answering "no contact" to every pair.
      sep_weight  a (B,L,L) multiplier from separation_weight(). Short-range vs
                  long-range -- stops the free diagonal band absorbing half the
                  gradient on positives.

    **min_sep defaults to 0 on purpose: train on ALL pairs.** The short band is
    part of the answer and the model has to be able to produce it. Restrict the
    METRIC to long range, not the loss; use sep_weight to shift emphasis instead.
    min_sep is exposed here only for the ablation that trains long-range-only, so
    that experiment does not need a second loss function.

    Padding is excluded rather than labelled zero: a padded pair is not a "no
    contact" observation, and at the padding fractions this repo runs the padded
    block would otherwise be most of the loss.
    """
    pm = pair_mask_for(mask, min_sep)
    if not pm.any():
        return logits.sum() * 0.0        # keeps the graph, contributes nothing
    per_pair = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none", pos_weight=pos_weight)
    w = pm.float() if sep_weight is None else pm.float() * sep_weight
    return (per_pair * w).sum() / w.sum().clamp_min(1.0)


def crop_pair(x, maps, mask, crop=CROP, generator=None):
    """Random contiguous crop to `crop` residues per side. Training only.

    Peak pair memory goes as crop^2, and this is the knob to turn on OOM -- not
    the protein count, which barely matters once the pair tensor dominates.

    A CONTIGUOUS crop, not a random subset: contact structure is a function of
    which residues are near in space, and a scattered subset would present the
    model with a chain that does not exist. The crop is aligned across all three
    tensors so pair (i,j) still refers to the same residues.
    """
    B, L, _ = x.shape
    if L <= crop:
        return x, maps, mask
    start = int(torch.randint(0, L - crop + 1, (1,), generator=generator).item())
    sl = slice(start, start + crop)
    return x[:, sl], maps[:, sl, sl], mask[:, sl]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L = 3, 48
    model = ContactPredictor().eval()
    x = torch.randn(B, L, IN_DIM)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 30:] = False        # a padded protein, to exercise the masking paths
    mask[2, 40:] = False

    n_par = sum(p.numel() for p in model.parameters())
    print(f"(a) parameters: {n_par/1e6:.2f}M   "
          f"(sequence encoder upstream is 43.1M)")

    with torch.no_grad():
        out = model(x, mask)
    assert out.shape == (B, L, L), out.shape
    print(f"    output shape {tuple(out.shape)}  OK")

    # ---- (b) symmetry ---------------------------------------------------
    asym = (out - out.transpose(1, 2)).abs().max().item()
    assert asym < 1e-5, asym
    print(f"(b) symmetry: max |M - M^T| = {asym:.2e}  OK")

    # ---- (c) padding cannot leak ----------------------------------------
    # Same protein, more padding. The real x real block must be IDENTICAL --
    # if it moves, padded residues are reaching real ones through attention.
    x2 = torch.cat([x, torch.randn(B, 20, IN_DIM)], dim=1)
    m2 = torch.cat([mask, torch.zeros(B, 20, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out2 = model(x2, m2)
    real = mask[0].sum()
    leak = (out[0, :real, :real] - out2[0, :real, :real]).abs().max().item()
    assert leak < 1e-4, f"padding leaked into real pairs: {leak}"
    print(f"(c) padding leak: max delta = {leak:.2e}  OK")

    # ---- (d) the output actually depends on the input --------------------
    # Cheap wiring check. It does NOT show the trained model uses the sequence --
    # only a trained run with use_relpos=False can say that.
    with torch.no_grad():
        out_zero = model(torch.zeros_like(x), mask)
    delta = (out - out_zero).abs().mean().item()
    assert delta > 1e-4, "output ignores its input"
    print(f"(d) input dependence: mean |delta| = {delta:.4f}  OK")

    # ---- (e) gradients reach every parameter -----------------------------
    model.train()
    tgt = (torch.rand(B, L, L) < 0.05).float()
    tgt = ((tgt + tgt.transpose(1, 2)) > 0).float()
    logits = model(x, mask)
    contact_loss(logits, tgt, mask, pos_weight=torch.tensor(20.0)).backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"no gradient reached: {dead}"
    print(f"(e) gradients reached all {len(list(model.parameters()))} tensors  OK")

    # ---- (f) both weights are measured off the data ----------------------
    # On a realistic map: ~40% of contacts sit at |i-j| <= 2, so long_weight comes
    # out well above 1. The random target here is uniform in |i-j| and so has no
    # short-range excess -- long_weight ~1 is the CORRECT answer for this input,
    # and the check is that it does not fabricate an imbalance that is not there.
    band = ((torch.arange(L)[:, None] - torch.arange(L)[None, :]).abs() <= 2)
    realistic = ((tgt + band.float().unsqueeze(0)) > 0).float()
    pw = pos_weight_from_maps(realistic, mask).item()
    lw_flat = long_weight_from_maps(tgt, mask).item()
    lw_real = long_weight_from_maps(realistic, mask).item()
    print(f"(f) pos_weight {pw:.1f} | long_weight: uniform target {lw_flat:.2f}, "
          f"with a diagonal band {lw_real:.2f}")
    assert lw_real > lw_flat, "long_weight did not react to a short-range excess"

    sw = separation_weight(mask, long_weight=lw_real)
    assert sw is not None and sw.shape == (B, L, L)
    assert separation_weight(mask, long_weight=1.0) is None, "1.0 should be a no-op"
    print(f"    separation_weight {tuple(sw.shape)}, "
          f"short band {sw[0, 0, 0]:.1f} / long {sw[0, 0, -1]:.1f}  OK")

    # ---- (g) it can actually fit something -------------------------------
    # The check that matters: drive a SINGLE protein's map into the model and see
    # the loss fall. If this does not move, nothing about the wiring above helps.
    torch.manual_seed(1)
    one_x = torch.randn(1, 40, IN_DIM)
    one_m = torch.ones(1, 40, dtype=torch.bool)
    idx = torch.arange(40)
    one_t = ((idx[:, None] - idx[None, :]).abs() <= 2).float().unsqueeze(0)
    one_t[0, 5, 30] = one_t[0, 30, 5] = 1.0      # one long-range contact to find

    small = ContactPredictor(n_pair_blocks=2).train()
    opt = torch.optim.Adam(small.parameters(), lr=3e-3)
    pw = pos_weight_from_maps(one_t, one_m)
    first = last = None
    for step in range(150):
        opt.zero_grad()
        loss = contact_loss(small(one_x, one_m), one_t, one_m, pos_weight=pw)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < 0.5 * first, f"cannot overfit one protein: {first:.3f} -> {last:.3f}"
    small.eval()
    with torch.no_grad():
        got = (small(one_x, one_m)[0, 5, 30] > 0).item()
    print(f"(g) overfit one protein: loss {first:.3f} -> {last:.3f}, "
          f"long-range contact recovered: {bool(got)}  OK")

    # ---- (h) crop keeps the three tensors aligned -------------------------
    cx, cm_maps, cmask = crop_pair(torch.randn(2, 300, IN_DIM),
                                   torch.rand(2, 300, 300),
                                   torch.ones(2, 300, dtype=torch.bool), crop=64)
    assert cx.shape[1] == cm_maps.shape[1] == cm_maps.shape[2] == cmask.shape[1] == 64
    print(f"(h) crop: 300 -> {cx.shape[1]} residues, maps {tuple(cm_maps.shape)}  OK")

    # ---- memory, since it is the thing that will actually stop you --------
    print("\npair-tensor memory, ONE activation at PAIR_DIM=64, fp32:")
    for L_ in (128, 192, 256, 512, 990):
        mb = L_ * L_ * PAIR_DIM * 4 / 1e6
        print(f"  L={L_:<5} {mb:8.1f} MB/protein   "
              f"(a block holds ~4 of these; train with CROP={CROP})")
