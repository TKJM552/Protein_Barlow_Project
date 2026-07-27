"""Standalone diagnostics for the protein JEPA.

Purpose: let you SEE whether the model actually learns, independent of the unit
tests. Every check prints numbers next to the value you should expect if things
are working and if they're broken -- read the output and judge for yourself.

It imports the existing modules (data pipeline, SequenceEncoder, Predictor,
MapEncoder, BarlowTwinsLoss) via train.py and does NOT reimplement them.

Usage:
    python eval.py --test overfit|shuffled|collapse|retrieval|probe|all
                   [--ckpt checkpoints/best.pt] [--seed 0]

    # any of train.py's environment flags work here too
    python eval.py --test probe --ckpt $CKPT_DIR/best.pt --device cuda

Without --ckpt the model is randomly initialized. Several tests (shuffled, probe
baselines) need random init as a reference, so a random-init run is meaningful --
it should show the "broken/near-chance" numbers, which confirms the diagnostics
are calibrated.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import train
from seq_encoder import ProteinSequenceDataset, collate_pad, BATCH_SIZE

# Bound at import to train.py's defaults, then re-read from train after
# apply_cli_overrides() runs in main() -- see refresh_from_train().
DEVICE = train.DEVICE
DATA_DIR = train.DATA_DIR

# Number of residues every CKA ranking comparison is computed over. CKA is
# inflated when estimated from fewer samples, so comparisons of different
# lengths are not rankable against each other -- a short decoy would beat the
# true structure. Fixing the length makes every comparison like for like.
# (The dataset median length is 218, so most proteins qualify.)
CMP_LEN = 150


def refresh_from_train():
    """Re-read the values train.apply_cli_overrides() may have just changed.

    train.DEVICE / train.DATA_DIR are rebound by the override pass, so the copies
    captured at import above would otherwise be stale for the rest of this run.
    """
    global DEVICE, DATA_DIR, BATCH_SIZE
    DEVICE = train.DEVICE
    DATA_DIR = train.DATA_DIR
    BATCH_SIZE = train.BATCH_SIZE


# ===========================================================================
# Shared setup / helpers
# ===========================================================================
def build_setup(ckpt, seed):
    """Seed everything, build the model, optionally load a checkpoint."""
    train.set_seed(seed)
    modules = train.build_modules()
    if ckpt:
        epoch, val = train.load_checkpoint(ckpt, modules)
        status = f"checkpoint '{ckpt}' (epoch {epoch}, val {val:.3f})"
    else:
        status = "RANDOM INIT (no checkpoint) -- expect broken/near-chance numbers"
    return modules, status


def dataset():
    return ProteinSequenceDataset(DATA_DIR)


def set_dropout_zero(modules):
    for m in modules.values():
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0


def all_params(modules):
    return [p for m in modules.values() for p in m.parameters()]


def encode_batch(modules, batch):
    """Run the full forward and return the intermediate reps + mask + maps."""
    padded_ints, mask, padded_maps = train.to_device(batch)
    seq_repr, _ = modules["sequence_encoder"](padded_ints, mask)
    pred, _ = modules["predictor"](seq_repr, mask)
    target, _ = modules["map_encoder"](padded_maps, mask)
    return seq_repr, pred, target, mask, padded_maps


def cycle(loader):
    while True:
        for b in loader:
            yield b


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ===========================================================================
# TEST 1 -- overfit_tiny
# ===========================================================================
def overfit_tiny(modules, steps=300, seed=0):
    hr("TEST 1  overfit_tiny  -- can the model memorize 4 proteins?")
    print("Judged on ON_DIAG (invariance), not total loss. Total loss is dominated by")
    print("lambda*off_diag, and with only 4 proteins the residue count N is far below")
    print("EXPANDER_DIM, so the cross-correlation is rank-deficient and off_diag has a")
    print("floor it CANNOT reach zero from. Thresholding the total therefore reports")
    print("'PARTIAL' even when gradients are flowing perfectly.\n")
    print("Expected HEALTHY : on_diag drops steeply (>~70%).")
    print("Expected BROKEN  : on_diag plateaus near its start -> gradients not flowing.\n")

    ds = dataset()
    loader = DataLoader(Subset(ds, list(range(4))), batch_size=4,
                        shuffle=False, collate_fn=collate_pad)
    batch = train.to_device(next(iter(loader)))

    set_dropout_zero(modules)          # remove noise so memorization is clean
    train.set_mode(modules, train=True)
    opt = torch.optim.AdamW(all_params(modules), lr=train.LR,
                            weight_decay=train.WEIGHT_DECAY)

    losses, ons = [], []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, on_d, off_d = train.forward_loss(modules, batch, use_amp=False)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        ons.append(on_d.item())
        if step % 25 == 0 or step == steps - 1:
            print(f"  step {step:3d}: loss {loss.item():10.3f}  (on {on_d.item():.3f}, off {off_d.item():.1f})")

    # Total loss is reported for context, but the VERDICT is on on_diag.
    t_first, t_last = losses[0], min(losses[-1], min(losses))
    first, last = ons[0], min(ons[-1], min(ons))
    drop = (first - last) / max(abs(first), 1e-9)
    t_drop = (t_first - t_last) / max(abs(t_first), 1e-9)
    print(f"\n  total loss {t_first:.3f} -> {t_last:.3f}   ({100*t_drop:.1f}% drop, "
          f"floored by off_diag -- ignore)")
    print(f"  ON_DIAG    {first:.3f} -> {last:.3f}   ({100*drop:.1f}% drop)  <- the verdict")
    if drop > 0.7:
        print("  -> HEALTHY: gradients flow and the model can fit a tiny set.")
    elif drop > 0.2:
        print("  -> PARTIAL: it moves but not far; try more steps or a higher LR.")
    else:
        print("  -> WARNING: loss barely moved. Likely causes:")
        print("     * a module missing from the optimizer's param list,")
        print("     * a masking bug zeroing gradients, or")
        print("     * learning rate too low.")


# ===========================================================================
# TEST 2 -- shuffled_control
# ===========================================================================
def shuffled_control(ckpt, seed, steps=500, n_prot=200):
    hr("TEST 2  shuffled_control  -- does REAL sequence<->map pairing beat a mismatch?")
    print("Two runs, same init/seed/data order, only the pairing differs.")
    print("Expected HEALTHY : REAL final loss clearly LOWER than SHUFFLED.")
    print("Expected BROKEN  : the two are within noise -> not learning seq->structure.\n")
    print("Mask handling: we roll padded_maps by 1 along the batch dim and KEEP the")
    print("batch's real-residue mask. Lengths differ, so the rolled map is simply a")
    print("mismatched structure under protein i's mask (extra rows masked out; short")
    print("maps leave real residues contact-less). That mismatch is the whole point.\n")

    def run(shuffled):
        modules, _ = build_setup(ckpt, seed)      # identical init both times
        set_dropout_zero(modules)                 # clean comparison, no RNG divergence
        train.set_mode(modules, train=True)
        opt = torch.optim.AdamW(all_params(modules), lr=train.LR,
                                weight_decay=train.WEIGHT_DECAY)
        gen = torch.Generator().manual_seed(seed)  # same batch order both runs
        loader = DataLoader(Subset(dataset(), list(range(n_prot))),
                            batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_pad, generator=gen)
        it = cycle(loader)
        recent = []
        for step in range(steps):
            padded_ints, mask, padded_maps = train.to_device(next(it))
            if shuffled:
                padded_maps = torch.roll(padded_maps, shifts=1, dims=0)  # mismatch pairing
                # A sequence may now be paired with a SHORTER protein's map, whose
                # 1-diagonal doesn't cover this protein's real residues -> that breaks
                # the map encoder's "diagonal==1 at real positions" contract. Restore
                # the self-contact diagonal (i==i is 1 for every protein, so this
                # leaks NO pairing info); the mismatched OFF-diagonal contacts, which
                # are the whole point of the control, stay untouched.
                L = padded_maps.shape[1]
                d = torch.arange(L, device=padded_maps.device)
                padded_maps = padded_maps.clone()
                padded_maps[:, d, d] = mask.float()
            opt.zero_grad(set_to_none=True)
            loss, _, _ = train.forward_loss(modules, (padded_ints, mask, padded_maps), use_amp=False)
            loss.backward()
            opt.step()
            if step >= steps - 20:
                recent.append(loss.item())
        return float(np.mean(recent))

    real_loss = run(shuffled=False)
    shuf_loss = run(shuffled=True)
    gap = shuf_loss - real_loss
    print(f"  REAL     final loss (last-20 avg): {real_loss:.3f}")
    print(f"  SHUFFLED final loss (last-20 avg): {shuf_loss:.3f}")
    print(f"  gap (shuffled - real)            : {gap:+.3f}")
    if gap > 0.05 * abs(real_loss) and gap > 1.0:
        print("  -> HEALTHY: real pairing is genuinely easier; the model uses the sequence.")
    else:
        print("  -> WARNING: real and shuffled are within noise. The model is NOT")
        print("     learning sequence -> structure (this is the key diagnostic here).")


# ===========================================================================
# TEST 3 -- collapse_diagnostics
# ===========================================================================
def _rep_stats(name, Xr, seed=0):
    N, D = Xr.shape
    std = Xr.std(0)
    n_dead = int((std < 1e-3).sum())
    print(f"  [{name}] N={N} residues, D={D}")
    print(f"    per-dim std   : min {std.min():.4f}  median {std.median():.4f}  max {std.max():.4f}")
    print(f"    dims std<1e-3 : {n_dead}   (HEALTHY: few/none | COLLAPSED: many)")

    S = torch.linalg.svdvals(Xr.float())
    n_sv = int((S > 0.01 * S[0]).sum())
    pr = (S.sum() ** 2) / (S.pow(2).sum())
    print(f"    eff. rank     : {n_sv} sing.vals >1% of max | participation-ratio {pr:.1f}")
    print(f"                    (HEALTHY: hundreds | DEGENERATE: single digits)")

    g = torch.Generator().manual_seed(seed)
    k = min(5000, N * (N - 1) // 2)
    i = torch.randint(0, N, (k,), generator=g)
    j = torch.randint(0, N, (k,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    Xn = Xr / Xr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = (Xn[i] * Xn[j]).sum(1).mean().item()
    print(f"    mean |cos| of random pairs: {cos:+.4f}   (HEALTHY: ~0 | COLLAPSED: ~1)")


def collapse_diagnostics(modules, seed=0):
    hr("TEST 3  collapse_diagnostics  -- are the representations degenerate?")
    train.set_mode(modules, train=False)
    loader = DataLoader(dataset(), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_pad)
    with torch.no_grad():
        batch = next(iter(loader))
        _, pred, target, mask, _ = encode_batch(modules, batch)
        loss, on_d, off_d = modules["expanders"](pred, target, mask)
        pred_r = pred[mask].cpu()
        target_r = target[mask].cpu()

    _rep_stats("pred  ", pred_r, seed)
    print()
    _rep_stats("target", target_r, seed)

    print(f"\n  Barlow Twins terms (separately):")
    print(f"    on_diag  {on_d.item():10.3f}   invariance -- high = pred doesn't match target")
    print(f"    off_diag {off_d.item():10.1f}   redundancy -- high = dimensions correlated (collapse)")
    print("    Healthy training drives BOTH down; a large residual off_diag with tiny")
    print("    on_diag means the two views agree but are collapsing onto few dimensions.")


# ===========================================================================
# TEST 4 -- retrieval_accuracy
# ===========================================================================
def retrieval_accuracy(modules, seed, n_prot=25):
    """Can we match each protein's sequence to ITS OWN structure, among distractors?

    HISTORY -- this test used to mean-pool pred and target into one vector per
    protein and rank them by cosine. That reported ~4% (below chance) and the
    conclusion "nothing protein-specific learned" was WRONG. Two separate flaws:

      1. Mean-pooling destroys the signal. After averaging, different proteins'
         target vectors sit at ~0.99 cosine to each other -- indistinguishable.
         The protein's identity lives in how its residues are ARRANGED relative
         to one another, and averaging is exactly the operation that discards it.
      2. Cosine assumes a shared basis. Barlow Twins pushes pred and target
         through SEPARATE expanders and only correlates dimensions, so nothing
         ever forces the two 512-d spaces into a common frame.

    CKA has neither problem: it compares the arrangement of residues and is
    invariant to rotation. Same model, same checkpoint, 4% -> 100%.

    The old pooled-cosine number is still printed, as a standing warning that
    anything downstream which mean-pools these embeddings will lose the signal.
    """
    hr("TEST 4  retrieval_accuracy  -- is each protein's pred closest to ITS target?")
    chance = 100.0 / n_prot
    print(f"  {n_prot} proteins, each matched against all {n_prot} structures.")
    print(f"  Chance top-1 = 1/{n_prot} = {chance:.1f}%")
    print("  Expected HEALTHY : far above chance -> the mapping is protein-specific.")
    print("  Expected BROKEN  : near chance -> nothing protein-specific learned.\n")

    train.set_mode(modules, train=False)
    ds = dataset()
    g = torch.Generator().manual_seed(seed)

    # Every CKA in the ranking must be computed over the SAME number of residues:
    # fewer samples inflates CKA, so mixing lengths would let short decoys beat
    # the true structure. Restrict to proteins at least CMP_LEN long, then compare
    # exactly the first CMP_LEN residues of each.
    # Break as soon as we have enough -- a list comprehension over the full
    # permutation would decompress every .npz in the dataset.
    idx = []
    for i in torch.randperm(len(ds), generator=g).tolist():
        if len(ds[i][0]) >= CMP_LEN:
            idx.append(i)
            if len(idx) >= n_prot:
                break
    if len(idx) < 2:
        print(f"  too few proteins of length >= {CMP_LEN}; skipping.")
        return

    reps = []
    with torch.no_grad():
        for i in idx:
            seq_ints, cmap = ds[i]
            L = seq_ints.shape[0]
            ints = seq_ints.unsqueeze(0).to(DEVICE)
            maps = cmap.unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
            seq_repr, _ = modules["sequence_encoder"](ints, mask)
            pred, _ = modules["predictor"](seq_repr, mask)
            target, _ = modules["map_encoder"](maps, mask)
            reps.append((pred[0].float(), target[0].float()))

    # --- primary: CKA similarity (no pooling, rotation-invariant) ---------
    N = len(reps)
    sim = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            sim[a, b] = linear_cka(reps[a][0][:CMP_LEN], reps[b][1][:CMP_LEN])

    gold = np.arange(N)
    ranks = (sim >= sim[gold, gold][:, None]).sum(1)
    top1 = 100 * float((sim.argmax(1) == gold).mean())
    top3 = 100 * float((ranks <= 3).mean())

    print(f"  top-1 accuracy : {top1:.1f}%   (chance {chance:.1f}%)")
    print(f"  top-3 accuracy : {top3:.1f}%")
    print(f"  median rank    : {int(np.median(ranks))}/{N}   (chance {N // 2})")

    # --- secondary: the old pooled-cosine readout, for contrast -----------
    P = torch.stack([p.mean(0) for p, _ in reps])
    T = torch.stack([t.mean(0) for _, t in reps])
    Pn = P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-8)
    pooled_top1 = 100 * float((( Pn @ Tn.T).argmax(1).numpy() == gold).mean())
    off = ~torch.eye(N, dtype=torch.bool)
    print(f"\n  for contrast, the OLD mean-pooled cosine readout: {pooled_top1:.1f}%")
    print(f"  (different proteins' pooled targets sit at cosine "
          f"{(Tn @ Tn.T)[off].mean():.3f} to each other --")
    print("   pooling collapses them together, which is why it cannot tell them apart)")

    if top1 > 3 * chance:
        print("\n  -> HEALTHY: the sequence->structure mapping is strongly protein-specific.")
        if pooled_top1 < 3 * chance:
            print("     NOTE: do NOT mean-pool these embeddings downstream. The signal is")
            print("     in the residue ARRANGEMENT; averaging destroys it, as shown above.")
    else:
        print("\n  -> WARNING: near chance even under CKA; nothing protein-specific learned.")


# ===========================================================================
# TEST 5 -- linear_probe
# ===========================================================================
def _protein_repr(enc, ds, idx):
    """(L,512) sequence reps + (L,L) ground-truth contact map for one protein."""
    seq_ints, cmap = ds[idx]
    L = seq_ints.shape[0]
    ints = seq_ints.unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        rep, _ = enc(ints, mask)
    return rep[0].cpu(), cmap.cpu()


def _pair_feat(V, i, j, mode):
    """Symmetric pair features. encoder: [v_i+v_j, v_i*v_j] (1024). distance: |i-j|."""
    if mode == "distance":
        return (i - j).abs().float().unsqueeze(1)            # (P, 1)
    vi, vj = V[i], V[j]
    return torch.cat([vi + vj, vi * vj], dim=1)              # (P, 1024) -- symmetric


def _sample_train_pairs(enc, ds, idx_list, mode, pos_cap=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for idx in idx_list:
        V, cmap = _protein_repr(enc, ds, idx) if mode != "distance" else (None, ds[idx][1].cpu())
        L = cmap.shape[0]
        iu, ju = torch.triu_indices(L, L, offset=1)          # i<j pairs, no diagonal
        lab = cmap[iu, ju]
        pos = torch.where(lab == 1)[0]
        neg = torch.where(lab == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        take = min(pos_cap, len(pos))
        pos = pos[torch.randperm(len(pos), generator=g)[:take]]
        neg = neg[torch.randperm(len(neg), generator=g)[:take]]  # balanced negatives
        sel = torch.cat([pos, neg])
        i, j = iu[sel], ju[sel]
        feats.append(_pair_feat(V, i, j, mode))
        labels.append(lab[sel])
    return torch.cat(feats), torch.cat(labels)


def _train_logreg(X, y, in_dim, steps=400, seed=0):
    """Standardize features, then fit a single Linear via BCE (logistic regression)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xn = (X - mu) / sd
    probe = nn.Linear(in_dim, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        out = probe(Xn).squeeze(1)
        lossf(out, y.float()).backward()
        opt.step()
    return probe, mu, sd


def _auc(scores, labels):
    """AUC via the Mann-Whitney U statistic (no sklearn needed)."""
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return ((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def _evaluate_probe(enc, probe, mu, sd, ds, idx_list, mode, max_pairs=60000, seed=0):
    """Long-range (|i-j|>12) precision@L/5, averaged over test proteins, plus pooled AUC."""
    g = torch.Generator().manual_seed(seed)
    precisions, all_scores, all_labels = [], [], []
    for idx in idx_list:
        V, cmap = _protein_repr(enc, ds, idx) if mode != "distance" else (None, ds[idx][1].cpu())
        L = cmap.shape[0]
        iu, ju = torch.triu_indices(L, L, offset=1)
        lr = (ju - iu) > 12                                   # LONG-RANGE only
        iu, ju = iu[lr], ju[lr]
        if len(iu) < 5:
            continue
        if len(iu) > max_pairs:                               # cap for tractability
            keep = torch.randperm(len(iu), generator=g)[:max_pairs]
            iu, ju = iu[keep], ju[keep]
        feat = (_pair_feat(V, iu, ju, mode) - mu) / sd
        with torch.no_grad():
            scores = probe(feat).squeeze(1)
        labels = cmap[iu, ju]
        k = max(1, L // 5)
        top = scores.topk(min(k, len(scores))).indices
        precisions.append(labels[top].float().mean().item())
        all_scores.append(scores)
        all_labels.append(labels)
    auc = _auc(torch.cat(all_scores), torch.cat(all_labels))
    return float(np.mean(precisions)), auc


def linear_probe(modules, seed, n_train=60, n_test=30):
    hr("TEST 5  linear_probe  -- does the FROZEN encoder contain contact information?")
    print("Precision@L/5 on LONG-RANGE (|i-j|>12) contacts -- the field-standard metric.")
    print("Expected: trained encoder > random encoder > distance-only.")
    print("Verdict rules: if trained <= random, JEPA added nothing; if trained does not")
    print("beat distance-only on long-range, it only learned the trivial 'near in")
    print("sequence = near in space' prior, not real structure.\n")

    ds = dataset()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:n_train + n_test]

    trained_enc = modules["sequence_encoder"].eval()
    train.set_seed(seed + 999)
    random_enc = train.SequenceEncoder().to(DEVICE).eval()   # untrained baseline

    rows = []
    for label, enc, mode, in_dim in [
        ("trained encoder", trained_enc, "encoder", 1024),
        ("random encoder ", random_enc, "encoder", 1024),
        ("distance-only  ", None, "distance", 1),
    ]:
        Xtr, ytr = _sample_train_pairs(enc, ds, tr_idx, mode, seed=seed)
        probe, mu, sd = _train_logreg(Xtr, ytr, in_dim, seed=seed)
        p_at, auc = _evaluate_probe(enc, probe, mu, sd, ds, te_idx, mode, seed=seed)
        rows.append((label, p_at, auc))

    print(f"  {'probe':<18}{'P@L/5 (long-range)':<22}{'AUC':<8}")
    print("  " + "-" * 46)
    for label, p_at, auc in rows:
        print(f"  {label:<18}{p_at:<22.3f}{auc:<8.3f}")

    trained_p, random_p, dist_p = rows[0][1], rows[1][1], rows[2][1]
    print("\n  VERDICT:")
    if trained_p <= random_p + 1e-6:
        print("  * trained <= random  -> JEPA pretraining added NOTHING over random weights.")
    elif trained_p <= dist_p + 1e-6:
        print("  * trained > random but <= distance-only -> only the trivial sequence-")
        print("    proximity prior; no real long-range structure learned.")
    else:
        print("  * trained > random AND > distance-only -> the encoder genuinely captures")
        print("    long-range contact structure. This is the outcome you want.")


# ===========================================================================
# TEST 6 -- representational_alignment (CKA)
# ===========================================================================
def linear_cka(X, Y):
    """Linear CKA between two (N, D) representations of the SAME N samples.

    Rotation- and scale-invariant: it asks whether the two spaces induce the same
    geometry over the samples (which residues sit near which), NOT whether their
    coordinate axes agree. That is the right question here -- Barlow Twins pushes
    pred and target through SEPARATE expanders and only correlates dimensions, so
    nothing ever forces the two 512-d spaces into a shared basis. A plain cosine
    between pred_i and target_i therefore reads ~0 even when training worked.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (Y.T @ X).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return float(num / den.clamp_min(1e-12))


def representational_alignment(modules, seed, n_prot=15):
    hr("TEST 6  representational_alignment  -- do the two branches share a geometry?")
    print("Linear CKA between pred and target, per protein, vs an UNTRAINED model.")
    print("Expected HEALTHY : trained CKA high (>0.5) and far above random init.")
    print("Expected BROKEN  : trained ~= random -> training changed no geometry.\n")

    ds = dataset()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_prot].tolist()

    train.set_seed(seed + 999)
    random_modules = train.build_modules()      # untrained baseline
    for m in random_modules.values():
        m.eval()
    train.set_mode(modules, train=False)

    def cka_for(mods, i):
        seq_ints, cmap = ds[i]
        L = seq_ints.shape[0]
        ints = seq_ints.unsqueeze(0).to(DEVICE)
        maps = cmap.unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            seq_repr, _ = mods["sequence_encoder"](ints, mask)
            pred, _ = mods["predictor"](seq_repr, mask)
            target, _ = mods["map_encoder"](maps, mask)
        return linear_cka(pred[0].float(), target[0].float())

    tr_cka = np.array([cka_for(modules, i) for i in idx])
    rd_cka = np.array([cka_for(random_modules, i) for i in idx])

    print(f"  {'':16}{'mean':>8}{'min':>8}{'max':>8}")
    print("  " + "-" * 40)
    print(f"  {'trained':16}{tr_cka.mean():>8.3f}{tr_cka.min():>8.3f}{tr_cka.max():>8.3f}")
    print(f"  {'random init':16}{rd_cka.mean():>8.3f}{rd_cka.min():>8.3f}{rd_cka.max():>8.3f}")
    wins = int((tr_cka > rd_cka).sum())
    print(f"\n  trained higher on {wins}/{n_prot} proteins   "
          f"(mean gain {tr_cka.mean() - rd_cka.mean():+.3f})")

    if tr_cka.mean() > 0.5 and tr_cka.mean() > rd_cka.mean() + 0.2:
        print("  -> HEALTHY: the sequence branch arranges a protein's residues much like")
        print("     the structure branch does. Note this is a WITHIN-protein measure;")
        print("     TEST 4 uses the same quantity ACROSS proteins to check it is specific.")
    elif tr_cka.mean() > rd_cka.mean() + 0.05:
        print("  -> PARTIAL: some shared geometry, but not a decisive margin.")
    else:
        print("  -> WARNING: training did not align the two representation spaces.")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Protein JEPA diagnostics")
    ap.add_argument("--test", required=True,
                    choices=["overfit", "shuffled", "collapse", "retrieval", "probe",
                             "alignment", "all"])
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: random init)")
    ap.add_argument("--seed", type=int, default=0)
    train.add_override_args(ap)   # --data-dir / --ckpt-dir / --device / --batch-size / ...
    args = ap.parse_args()

    # Point train.py's globals at the requested dirs/device, then pick up the
    # results here. Must happen before anything builds a dataset or a module.
    train.apply_cli_overrides(args)
    refresh_from_train()

    modules, status = build_setup(args.ckpt, args.seed)
    print(f"device: {DEVICE} | model: {status} | seed: {args.seed}")

    t = args.test
    # overfit / shuffled TRAIN and mutate weights, so give each its own fresh model.
    if t in ("overfit", "all"):
        m, _ = build_setup(args.ckpt, args.seed)
        overfit_tiny(m, seed=args.seed)
    if t in ("shuffled", "all"):
        shuffled_control(args.ckpt, args.seed)
    if t in ("collapse", "all"):
        m, _ = build_setup(args.ckpt, args.seed)
        collapse_diagnostics(m, seed=args.seed)
    if t in ("retrieval", "all"):
        m, _ = build_setup(args.ckpt, args.seed)
        retrieval_accuracy(m, seed=args.seed)
    if t in ("probe", "all"):
        m, _ = build_setup(args.ckpt, args.seed)
        linear_probe(m, seed=args.seed)
    if t in ("alignment", "all"):
        m, _ = build_setup(args.ckpt, args.seed)
        representational_alignment(m, seed=args.seed)


if __name__ == "__main__":
    main()
