"""Barlow Twins loss for Protein Barlow.

Compares two per-residue representations and returns a scalar loss. Preventing
collapse IS its job: there is NO stop-gradient and NO EMA anywhere -- the
off-diagonal redundancy-reduction term is the anti-collapse mechanism, and BOTH
inputs (sequence-encoder output and map-encoder output) receive gradients.

The two inputs are symmetric -- neither predicts the other, neither is a target --
so they are named z_seq and z_map rather than pred/target.

Framing: Barlow Twins operates on a batch of VECTORS. Here each REAL RESIDUE is
one sample, so N = total non-padding residues across the whole batch (16 proteins
x their lengths = thousands), which is plenty for stable correlation estimates.

NOTE on conditioning: N should exceed EXPANDER_DIM for the DxD cross-correlation
to be well-conditioned in general. With thousands of residue-samples and
EXPANDER_DIM=2048 this holds comfortably; flag it if batches are ever made tiny
(the loss still computes -- it's just a rank-deficient estimate). train.py's
residue-budget batching keeps N near-constant at RESIDUES_PER_BATCH for exactly
this reason.

PRECISION: the loss is computed in fp32 even under autocast -- see
barlow_twins_core. This is not optional; fp16 cannot represent off_diag.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (named, easy to change)
# ---------------------------------------------------------------------------
REP_DIM = 512          # D of z_seq / z_map (both encoders already match)
EXPANDER_DIM = 2048    # width the loss is actually computed at
LAMBDA_OFFDIAG = 5e-3  # weight on the off-diagonal (redundancy) term; BT default range
EPS = 1e-5             # standardization denominator floor


def _off_diagonal(x):
    """Return a flat view of all off-diagonal elements of a square matrix x."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_core(za, zb, lambda_offdiag=LAMBDA_OFFDIAG, eps=EPS):
    """Steps 3-5 on two (N, D) sample matrices. Kept separate from the expander
    so it can be tested directly on constructed inputs.

    Runs in fp32 with autocast explicitly DISABLED, even when the encoders around
    it are running in fp16/bf16. This is a correctness requirement, not a
    preference:

        off_diag is a sum of D*(D-1) = 4.19M squared correlations. fp16 tops out
        at 65504, and a real 50-epoch run of this model reached off_diag = 89642
        during epoch 2 (see FINDINGS.md). Computed in fp16 that reduction returns
        `inf`, train.py's non-finite guard then skips EVERY optimizer step, and
        training silently does nothing but print "frequent skips".

    The cost is the (D, N) x (N, D) matmul running at fp32 rather than half
    precision -- ~15% of the forward FLOPs. Cheap insurance, and it is what every
    reference Barlow Twins/VICReg implementation does.

    Returns (loss, on_diag_detached, off_diag_detached).
    """
    N, _ = za.shape

    with torch.autocast(device_type=za.device.type, enabled=False):
        za = za.float()
        zb = zb.float()

        # 3. Standardize each dimension across the N samples. Population std
        #    (unbiased=False) is used deliberately: it matches the batch-norm-along-
        #    batch step in the BT paper and makes the diagonal of c land exactly at 1
        #    when the two views agree, so `on_diag` is a clean invariance signal.
        za = (za - za.mean(0)) / (za.std(0, unbiased=False) + eps)
        zb = (zb - zb.mean(0)) / (zb.std(0, unbiased=False) + eps)

        # 4. Empirical cross-correlation matrix (D, D); entries lie ~[-1, 1].
        c = (za.T @ zb) / N

        # 5. Loss.
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()   # invariance: the two views agree
        off_diag = _off_diagonal(c).pow(2).sum()         # redundancy reduction: anti-collapse
        loss = on_diag + lambda_offdiag * off_diag

    return loss, on_diag.detach(), off_diag.detach()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, rep_dim=REP_DIM, expander_dim=EXPANDER_DIM,
                 lambda_offdiag=LAMBDA_OFFDIAG, eps=EPS, shared_expander=False):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag
        self.eps = eps

        # 1. Expander: a small per-residue MLP REP_DIM -> EXPANDER_DIM. These are
        #    TRAINED THEN DISCARDED -- the representations we keep for downstream
        #    use are the PRE-expander ones (z_seq / z_map); the expander only
        #    shapes the space the loss is measured in. Default = two SEPARATE
        #    expanders (flip shared_expander=True to tie them).
        self.expander_a = self._build_expander(rep_dim, expander_dim)
        self.expander_b = (
            self.expander_a if shared_expander
            else self._build_expander(rep_dim, expander_dim)
        )

    @staticmethod
    def _build_expander(rep_dim, expander_dim):
        return nn.Sequential(
            nn.Linear(rep_dim, expander_dim),
            nn.BatchNorm1d(expander_dim),
            nn.ReLU(),
            nn.Linear(expander_dim, expander_dim),
        )

    def forward(self, z_seq, z_map, mask):
        """z_seq, z_map: (B, L, D). mask: (B, L) bool, True = real residue.

        z_seq comes from the sequence encoder, z_map from the map encoder. The loss
        is symmetric in the two, so the argument ORDER is a labelling convention
        only -- it decides which expander sees which branch, nothing more.
        """
        B, L, D = z_seq.shape
        assert z_map.shape == z_seq.shape, "z_seq and z_map must have the same shape"
        assert mask.shape == (B, L) and mask.dtype == torch.bool, "mask must be (B, L) bool"

        # 2. Flatten to samples and keep only REAL residues. We select BEFORE the
        #    expander on purpose: this keeps padding rows out of the expander's
        #    BatchNorm statistics AND out of the step-3 standardization, so the
        #    correlation is estimated over genuine residues only.
        flat_mask = mask.reshape(B * L)
        seq_real = z_seq.reshape(B * L, D)[flat_mask]        # (N, REP_DIM)
        map_real = z_map.reshape(B * L, D)[flat_mask]        # (N, REP_DIM)
        N = seq_real.shape[0]
        assert N > 1, "need more than one real residue for BT statistics"

        # 1. Expander applied per-residue (to real residues only).
        za = self.expander_a(seq_real)      # (N, EXPANDER_DIM)
        zb = self.expander_b(map_real)      # (N, EXPANDER_DIM)

        # 3-5.
        return barlow_twins_core(za, zb, self.lambda_offdiag, self.eps)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # --- (a) Shape + gradient flow to BOTH sides --------------------------
    loss_fn = BarlowTwinsLoss()   # train mode (BatchNorm uses batch stats)
    B, L, D = 4, 10, REP_DIM
    z_seq = torch.randn(B, L, D, requires_grad=True)
    z_map = torch.randn(B, L, D, requires_grad=True)

    # Mixed real/padding mask, varying real length per protein (all >= 2 real).
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :8] = True
    mask[2, :3] = True
    mask[3, :10] = True

    loss, on_d, off_d = loss_fn(z_seq, z_map, mask)
    assert loss.dim() == 0 and torch.isfinite(loss), "loss must be a finite scalar"

    loss.backward()

    def _has_grad(module):
        grads = [p.grad for p in module.parameters()]
        return all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)

    assert _has_grad(loss_fn.expander_a), "expander_a got no gradient"
    assert _has_grad(loss_fn.expander_b), "expander_b got no gradient (stop-grad leak?)"
    # Both raw inputs receive gradients too -> neither side is stop-gradiented.
    assert z_seq.grad is not None and z_seq.grad.abs().sum() > 0, "z_seq received no gradient"
    assert z_map.grad is not None and z_map.grad.abs().sum() > 0, "z_map received no gradient"
    print(f"(a) N={int(mask.sum())} real residues; loss={loss.item():.4f} "
          f"(on={on_d.item():.4f}, off={off_d.item():.4f}); grads on BOTH sides.")

    # --- (b) Identity sanity: za == zb with orthonormal columns -> c = I --
    # Build a matrix whose (centered) columns are orthonormal, so standardization
    # yields c ~ identity: on_diag ~ 0 and off_diag ~ 0, hence loss ~ 0.
    Nb, Db = 4096, 512
    M = torch.randn(Nb, Db)
    M = M - M.mean(0)                 # centered columns -> live in the zero-mean subspace
    Q, _ = torch.linalg.qr(M)         # (Nb, Db), Q^T Q = I (orthonormal columns)
    loss_i, on_i, off_i = barlow_twins_core(Q, Q.clone())
    print(f"(b) identity: loss={loss_i.item():.2e} (on={on_i.item():.2e}, off={off_i.item():.2e})")
    assert torch.isfinite(loss_i)
    assert on_i < 1e-2 and off_i < 1e-2 and loss_i < 1e-2, "identity case should be ~0"

    # --- (c) Collapse sanity: the loss punishes collapse, stays finite -----
    Nc, Dc = 1000, 512
    # (c1) Every SAMPLE the same vector -> zero variance per column -> standardized
    #      to 0 -> c = 0 -> on_diag = D (large). Loss is high; EPS keeps it finite.
    same_sample = torch.randn(Dc).unsqueeze(0).repeat(Nc, 1)   # all rows identical
    loss_c, on_c, off_c = barlow_twins_core(same_sample, same_sample.clone())
    assert torch.isfinite(loss_c), "collapse must not produce NaN/inf"
    assert loss_c > 100, "collapsed (identical-sample) representation should score high"
    print(f"(c1) sample-collapse: loss={loss_c.item():.1f} (on={on_c.item():.1f}, off={off_c.item():.1f}), finite.")

    # (c2) Every DIM identical (redundant features) -> c = all-ones -> the
    #      OFF-DIAGONAL term explodes. This is the redundancy the BT off-diagonal
    #      term exists to punish.
    redundant = torch.randn(Nc, 1).repeat(1, Dc)   # all columns identical, rows vary
    loss_r, on_r, off_r = barlow_twins_core(redundant, redundant.clone())
    assert torch.isfinite(loss_r)
    assert off_r > 100, "redundant-dimension collapse should blow up the off-diagonal term"
    print(f"(c2) dim-redundancy: loss={loss_r.item():.1f} (on={on_r.item():.1f}, off={off_r.item():.1f}), off-diagonal large.")

    # --- (d) fp16 range: off_diag must survive values a real run produces ---
    # The archived 50-epoch run peaked at off_diag = 89642 (epoch 2). fp16's max
    # is 65504, so a loss computed in half precision returns inf there and every
    # optimizer step gets skipped. Build inputs that land in that range and check
    # the loss is finite with autocast ON -- this is the regression test for it.
    D_ = EXPANDER_DIM
    target = 89642.0
    v = (target / (D_ * D_ - D_)) ** 0.5      # per-entry |c_ij| giving that off_diag
    # Two rank-1-ish views whose cross-correlation is ~v off the diagonal.
    Nf = 4096
    base = torch.randn(Nf, 1)
    za_f = (1 - v) ** 0.5 * torch.randn(Nf, D_) + v ** 0.5 * base
    zb_f = (1 - v) ** 0.5 * torch.randn(Nf, D_) + v ** 0.5 * base
    with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
        loss_f, on_f, off_f = barlow_twins_core(za_f, zb_f)
    assert loss_f.dtype == torch.float32, f"loss must be fp32, got {loss_f.dtype}"
    assert torch.isfinite(loss_f), "loss overflowed under fp16 autocast"
    assert off_f > torch.finfo(torch.float16).max, (
        "test did not actually exceed the fp16 range -- it proves nothing")
    print(f"(d) fp16 autocast: off_diag={off_f.item():.0f} > fp16 max "
          f"{torch.finfo(torch.float16).max:.0f}, loss finite and fp32.")

    print("sanity check passed: finite scalar, grads on both sides, identity ~0, "
          "collapse punished, fp16-safe.")
