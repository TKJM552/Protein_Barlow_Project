"""Barlow Twins loss for the protein JEPA.

Compares two per-residue representations and returns a scalar loss. Preventing
collapse IS its job: there is NO stop-gradient and NO EMA anywhere -- the
off-diagonal redundancy-reduction term is the anti-collapse mechanism, and BOTH
inputs (predictor output and map-encoder output) receive gradients.

Framing: Barlow Twins operates on a batch of VECTORS. Here each REAL RESIDUE is
one sample, so N = total non-padding residues across the whole batch (16 proteins
x their lengths = thousands), which is plenty for stable correlation estimates.

NOTE on conditioning: N should exceed EXPANDER_DIM for the DxD cross-correlation
to be well-conditioned in general. With thousands of residue-samples and
EXPANDER_DIM=2048 this holds comfortably; flag it if batches are ever made tiny
(the loss still computes -- it's just a rank-deficient estimate).
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants (named, easy to change)
# ---------------------------------------------------------------------------
REP_DIM = 512          # D of pred / target (both sides already match)
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

    Returns (loss, on_diag_detached, off_diag_detached).
    """
    N, _ = za.shape

    # 3. Standardize each dimension across the N samples. Population std
    #    (unbiased=False) is used deliberately: it matches the batch-norm-along-
    #    batch step in the BT paper and makes the diagonal of c land exactly at 1
    #    when the two views agree, so `on_diag` is a clean invariance signal.
    za = (za - za.mean(0)) / (za.std(0, unbiased=False) + eps)
    zb = (zb - zb.mean(0)) / (zb.std(0, unbiased=False) + eps)

    # 4. Empirical cross-correlation matrix (D, D); entries lie ~[-1, 1].
    c = (za.T @ zb) / N

    # 5. Loss.
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()   # invariance: pred matches target
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
        #    use are the PRE-expander ones (pred / target); the expander only
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

    def forward(self, pred, target, mask):
        """pred, target: (B, L, D). mask: (B, L) bool, True = real residue."""
        B, L, D = pred.shape
        assert target.shape == pred.shape, "pred and target must have the same shape"
        assert mask.shape == (B, L) and mask.dtype == torch.bool, "mask must be (B, L) bool"

        # 2. Flatten to samples and keep only REAL residues. We select BEFORE the
        #    expander on purpose: this keeps padding rows out of the expander's
        #    BatchNorm statistics AND out of the step-3 standardization, so the
        #    correlation is estimated over genuine residues only.
        flat_mask = mask.reshape(B * L)
        pred_real = pred.reshape(B * L, D)[flat_mask]        # (N, REP_DIM)
        target_real = target.reshape(B * L, D)[flat_mask]    # (N, REP_DIM)
        N = pred_real.shape[0]
        assert N > 1, "need more than one real residue for BT statistics"

        # 1. Expander applied per-residue (to real residues only).
        za = self.expander_a(pred_real)      # (N, EXPANDER_DIM)
        zb = self.expander_b(target_real)    # (N, EXPANDER_DIM)

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
    pred = torch.randn(B, L, D, requires_grad=True)
    target = torch.randn(B, L, D, requires_grad=True)

    # Mixed real/padding mask, varying real length per protein (all >= 2 real).
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :8] = True
    mask[2, :3] = True
    mask[3, :10] = True

    loss, on_d, off_d = loss_fn(pred, target, mask)
    assert loss.dim() == 0 and torch.isfinite(loss), "loss must be a finite scalar"

    loss.backward()

    def _has_grad(module):
        grads = [p.grad for p in module.parameters()]
        return all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)

    assert _has_grad(loss_fn.expander_a), "expander_a got no gradient"
    assert _has_grad(loss_fn.expander_b), "expander_b got no gradient (stop-grad leak?)"
    # Both raw inputs receive gradients too -> neither side is stop-gradiented.
    assert pred.grad is not None and pred.grad.abs().sum() > 0, "pred received no gradient"
    assert target.grad is not None and target.grad.abs().sum() > 0, "target received no gradient"
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

    print("sanity check passed: finite scalar, grads on both sides, identity ~0, collapse punished.")
