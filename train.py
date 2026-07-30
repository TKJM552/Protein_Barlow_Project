"""Joint training loop for Protein Barlow.

Wires together the already-built components (imported, not reimplemented):

  DataLoader -> SequenceEncoder -> z_seq \
                                          Barlow Twins loss
                MapEncoder -----> z_map  /

Two encoders read the SAME protein through different windows -- one sees only the
amino-acid sequence, the other only the contact map -- and Barlow Twins is asked
to make the two representations agree dimension-for-dimension. Neither branch
predicts the other and neither is a "target": the objective is symmetric, so the
two are named z_seq and z_map.

This is JOINT training: ONE AdamW optimizer holds the parameters of the sequence
encoder, the map encoder, AND the Barlow Twins expanders. There is no frozen
branch, no stop-gradient, and no EMA -- collapse is prevented purely by the loss's
off-diagonal term (see barlow_twins.py).
"""

import os
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, random_split, Subset

# --- existing components (import & wire; do NOT reimplement) ---------------
from seq_encoder import (
    TokenEmbedding,
    RoPEEncoder,
    ProteinSequenceDataset,
    LengthBucketSampler,
    collate_pad,
    dataset_lengths,
    BATCH_SIZE,
    MIN_RESIDUES,
    PAD_IDX,
)
from map_encoder import ContactMapEncoder
from barlow_twins import BarlowTwinsLoss

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Anything that differs between a laptop and a GPU pod (paths, device, workers,
# amp dtype, batch size) comes from config.py, which reads the environment.
# apply_cli_overrides() below lets a CLI flag beat both. Everything else is a
# genuine hyperparameter and lives here.
DATA_DIR = config.DATA_DIR
CKPT_DIR = config.CKPT_DIR
DEVICE = config.resolve_device()
NUM_WORKERS = config.NUM_WORKERS
AMP_DTYPE = config.AMP_DTYPE
# Training batches are formed by residue budget (see LengthBucketSampler), so
# BATCH_SIZE above is NOT what governs a training step -- this is.
RESIDUES_PER_BATCH = config.RESIDUES_PER_BATCH

LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 50
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.1
SEED = 0

# --- the positional shortcut -----------------------------------------------
# THE FAILURE. Barlow Twins treats each RESIDUE as a sample, and both branches
# can work out a residue's index in its chain: RoPE is relative, but the offsets
# to the two chain ENDS pin the absolute index down, and the sequence branch has
# full attention, so it reads both distances everywhere. `z_seq = z_map = f(index)`
# then satisfies the objective with no chemistry learned at all -- every c_ii is
# 1, and a positional code is high-rank and decorrelated, so it walks straight
# through the off-diagonal term, which punishes DIMENSIONAL collapse and has
# nothing to say about a shortcut. It is an attractor the model can drift into at
# ANY epoch, and val loss FALLS as it drifts, so val loss will never warn you.
#
# THE FIX. Subtract, per residue INDEX, the mean over the proteins in the batch,
# before the loss. Note barlow_twins_core ALREADY subtracts a mean at this point
# (its standardization step); this only conditions that mean on the index instead
# of pooling every residue together. What it removes is whatever every protein
# shares at index i -- a fact about the index, not about any protein.
#
# What this does and does not do:
#   * f(index) alone centres to EXACTLY zero -> no variance -> cannot produce
#     c_ii = 1. That shortcut is dead, not merely discouraged.
#   * f(index) * g(protein) survives, because g varies. But the two branches can
#     only CORRELATE through g, i.e. by agreeing about this particular protein --
#     position modulates the amplitude and cannot manufacture agreement alone.
#     So relative structure (helix at +4, sheet pairings, contact order) is kept
#     and rewarded in full: it differs between proteins at the same index.
#   * position stays INFERABLE -- the spread of the deviations still varies along
#     the chain. The claim is only that position alone stops PAYING. Whether some
#     near-positional shortcut survives anyway is empirical, which is what
#     free_information() below is for.
#   * it does NOT touch the encoders' outputs, only the loss's copy of them, so
#     nothing downstream shifts.
#
# The removed part is not lost, it is factored out: z = m[index] + deviation,
# and m is a population average -- something you MEASURE, not something gradient
# descent has to learn. Run with --no-position-centering for the B arm of the
# ablation that checks it costs no real information (see POD_SETUP.md).
POSITION_CENTERING = True
MIN_PROTEINS_PER_INDEX = 2   # an index held by one protein centres to exactly
                             # zero, so those residues leave the loss instead of
                             # entering the statistics as fake all-zero samples

# --- free-information monitor ----------------------------------------------
# One number per epoch: the fraction of z_seq reproducible from POSITION AND
# LENGTH alone -- everything the model got without reading chemistry. Cells are
# (fractional-position bin, length bin), which subsumes absolute index (fixing
# length and fraction fixes the index) and, unlike absolute-index bins, stays
# evenly populated whatever the chain length. It therefore also catches the
# length-normalised shortcut f(index/length), which per-index centring does not
# fully remove.
#
# HIGH is bad. Read the TREND: biased upward by roughly (cells / residues),
# ~0.004 here, so it separates 0.03 from 0.6 and is not to be quoted to three
# decimals or compared across different bin counts. Same small-sample trap
# FINDINGS.md records for CKA. For scale, map_encoder.py measured z_map at random
# init at 0.033 with the relative seed.
PROBE_PROTEINS = 192       # held-out chains, strided across the length range
PROBE_CHUNK = 16           # padded together at a time, to cap probe memory
N_FRAC_BINS = 24
N_LEN_BINS = 8
CKPT_EVERY_EPOCHS = 5
LOG_EVERY = 25              # print a running train-loss line every N steps
RESUME_FROM = None         # --resume PATH to continue a run; None = fresh run
WARM_START = False         # --warm-start: allow --resume across an ARCHITECTURE
                           # change. Loads only the weights that still fit, drops
                           # the optimizer state, restarts the LR schedule.
USE_AMP = True             # --no-amp to force full fp32 (cpu runs ignore this)
KEEP_EPOCH_CKPTS = False   # --keep-epoch-ckpts: write epoch_NNN.pt every
                           # CKPT_EVERY_EPOCHS instead of rolling one last.pt.
                           # Off by default: 10 snapshots of a 43M-param model
                           # plus optimizer state is 5.2 GB, and only the newest
                           # is a useful resume point.

# --- debug-mode state (inert unless --debug is passed) ---------------------
DEBUG_MAX_PROTEINS = None   # None = use the whole dataset (normal runs)
CKPT_PREFIX = ""            # "debug_" in debug mode so throwaway checkpoints
                            # never overwrite real ones


def apply_cli_overrides(args):
    """Let explicit CLI flags beat the environment/config defaults.

    Only flags the user actually passed are applied (every override-able flag
    defaults to None), so an unflagged run keeps the config.py value. Must be
    called BEFORE main() builds the data, model, or optimizer.

    Values are read via getattr so eval.py, whose parser defines only the shared
    subset of these flags, can reuse this function unchanged.
    """
    global DEVICE, USE_AMP, WARM_START, KEEP_EPOCH_CKPTS, POSITION_CENTERING

    # argparse dest -> the module-level constant it overrides.
    for dest, constant in [
        ("data_dir", "DATA_DIR"), ("ckpt_dir", "CKPT_DIR"),
        ("num_workers", "NUM_WORKERS"), ("amp_dtype", "AMP_DTYPE"),
        ("batch_size", "BATCH_SIZE"), ("epochs", "EPOCHS"), ("lr", "LR"),
        ("weight_decay", "WEIGHT_DECAY"), ("warmup_steps", "WARMUP_STEPS"),
        ("seed", "SEED"), ("ckpt_every", "CKPT_EVERY_EPOCHS"),
        ("resume", "RESUME_FROM"),
        ("residues_per_batch", "RESIDUES_PER_BATCH"),
    ]:
        value = getattr(args, dest, None)
        if value is not None:
            globals()[constant] = value

    # These need more than a straight assignment.
    if getattr(args, "device", None) is not None:
        DEVICE = config.resolve_device(args.device)
    if getattr(args, "no_amp", False):
        USE_AMP = False
    if getattr(args, "warm_start", False):
        WARM_START = True
    if getattr(args, "keep_epoch_ckpts", False):
        KEEP_EPOCH_CKPTS = True
    if getattr(args, "no_position_centering", False):
        POSITION_CENTERING = False

    config.amp_dtype(AMP_DTYPE)   # validate the dtype now, not 500 steps in


def add_override_args(parser):
    """Register the flags apply_cli_overrides() consumes. Shared with eval.py."""
    g = parser.add_argument_group("environment (overrides config.py / env vars)")
    g.add_argument("--data-dir", default=None,
                   help="processed .npz dataset dir (env: DATA_DIR)")
    g.add_argument("--ckpt-dir", default=None,
                   help="checkpoint output dir (env: CKPT_DIR)")
    g.add_argument("--device", default=None,
                   help="cuda | cuda:0 | cpu | mps (env: DEVICE; default: cuda if available)")
    g.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker processes (env: NUM_WORKERS; try 4-8 on a GPU)")
    g.add_argument("--amp-dtype", default=None, choices=["fp16", "bf16"],
                   help="autocast dtype (env: AMP_DTYPE; bf16 needs no loss scaling, "
                        "prefer it on A100/H100)")
    g.add_argument("--batch-size", type=int, default=None,
                   help="proteins per batch for eval.py's diagnostics (env: "
                        "BATCH_SIZE). NOT used by training -- see "
                        "--residues-per-batch")
    g.add_argument("--residues-per-batch", type=int, default=None,
                   help="training batch budget in residues (env: "
                        "RESIDUES_PER_BATCH). Batches group proteins of similar "
                        "length and close at (proteins x longest chain); this is "
                        "the knob to lower on OOM. Keep it above ~2048")
    return g


def apply_debug_overrides():
    """Fast, GPU-free bug-catching pass over a tiny subset.

    Called ONLY when --debug is passed, and only AFTER the constants above are
    defined -- these four overrides plus the smaller dataset are the *only*
    differences from a real run. Every other code path (model construction,
    optimizer, loss, training loop, collate) is identical, so shape/mask/wiring
    bugs surface here exactly as they would on a GPU.
    """
    global DEVICE, EPOCHS, BATCH_SIZE, RESIDUES_PER_BATCH, DEBUG_MAX_PROTEINS, CKPT_PREFIX
    DEVICE = torch.device("cpu")   # torch.device (not the str "cpu") so DEVICE.type
                                   # still works for autocast / GradScaler / .to()
    EPOCHS = 2
    BATCH_SIZE = 4
    RESIDUES_PER_BATCH = 1024      # several small batches -> exercises padding/masking
    DEBUG_MAX_PROTEINS = 20
    CKPT_PREFIX = "debug_"
    print("*** DEBUG MODE: cpu, 20 proteins, 2 epochs, 1024 residues/batch -- "
          "bug-catching only, not a real run ***")


# ---------------------------------------------------------------------------
# Thin wrapper: compose the existing embedding + RoPE encoder into one module
# so it is a single parameter group. (Composition, not reimplementation.)
# ---------------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = TokenEmbedding()
        self.encoder = RoPEEncoder()

    def forward(self, padded_ints, mask):
        embedded, mask = self.embedding(padded_ints, mask)
        seq_repr, mask = self.encoder(embedded, mask)
        return seq_repr, mask


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_loaders():
    """Train/val loaders, both batched by residue budget rather than protein count.

    Batches group proteins of similar length (LengthBucketSampler), which is where
    ~50% of this model's training FLOPs used to go: collate_pad pads every protein
    up to the batch's longest member, so a random batch of 16 drawn from chains of
    40..990 residues did 2.41x the useful work. See the sampler's docstring for why
    the budget is counted in RESIDUES and not proteins.

    Shuffling now lives in the sampler (which re-forms batches every epoch), so the
    loaders take a batch_sampler and no shuffle= flag.
    """
    full = ProteinSequenceDataset(DATA_DIR)
    # Debug only: cap the dataset BEFORE the split so BOTH train and val draw from
    # the capped set. The split logic below is otherwise unchanged.
    if DEBUG_MAX_PROTEINS is not None:
        full = Subset(full, range(min(DEBUG_MAX_PROTEINS, len(full))))
    n_total = len(full)
    n_val = int(VAL_FRACTION * n_total)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full, [n_train, n_val], generator=gen)

    # pin_memory only helps (and is only valid) for CUDA host->device copies.
    # persistent_workers keeps the worker pool alive across all EPOCHS instead of
    # re-forking and re-scanning the dataset every epoch.
    loader_kwargs = dict(
        collate_fn=collate_pad,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=LengthBucketSampler(dataset_lengths(train_set),
                                          RESIDUES_PER_BATCH, shuffle=True, seed=SEED),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        # Validation is a measurement: fixed batches so the number is comparable
        # epoch to epoch.
        batch_sampler=LengthBucketSampler(dataset_lengths(val_set),
                                          RESIDUES_PER_BATCH, shuffle=False),
        **loader_kwargs,
    )
    return train_loader, val_loader, n_train, n_val


# ---------------------------------------------------------------------------
# Modules + optimizer + scheduler
# ---------------------------------------------------------------------------
def build_modules():
    # No prediction head between the sequence encoder and the loss: the expanders
    # inside BarlowTwinsLoss are already a per-residue MLP on each branch, so a
    # second one added nothing the loss could not express. Its removal means
    # sequence_encoder's OWN output is what the objective shapes -- which is also
    # the tensor load_sequence_encoder() hands to downstream code.
    return {
        "sequence_encoder": SequenceEncoder().to(DEVICE),
        "map_encoder": ContactMapEncoder().to(DEVICE),
        "expanders": BarlowTwinsLoss().to(DEVICE),   # the loss module owns the expanders
    }


def build_optimizer(modules, total_steps):
    # ONE optimizer over ALL parameters: sequence encoder + map encoder + expanders.
    param_list = []
    for name, m in modules.items():
        params = list(m.parameters())
        assert len(params) > 0, f"module {name!r} contributed zero parameters"
        param_list += params
    assert len(param_list) > 0, "optimizer received no parameters"

    # Weight decay on matrices only. Every 1-D parameter here is a bias or a
    # LayerNorm/BatchNorm gain -- shrinking those toward zero is not
    # regularisation, it drags the normalisation statistics the network relies on.
    # (Embeddings are 2-D and keep decay, the usual convention.)
    decay = [p for p in param_list if p.ndim >= 2]
    no_decay = [p for p in param_list if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
    )

    # Linear warmup for WARMUP_STEPS, then cosine decay over the remaining steps.
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, param_list


# ---------------------------------------------------------------------------
# Forward data flow (shared by smoke test, training, and validation)
# ---------------------------------------------------------------------------
def to_device(batch):
    padded_ints, mask, padded_maps = batch
    # non_blocking pairs with the loaders' pin_memory: the H2D copy overlaps with
    # the previous batch's compute instead of stalling the GPU. No-op elsewhere.
    nb = DEVICE.type == "cuda"
    return (padded_ints.to(DEVICE, non_blocking=nb),
            mask.to(DEVICE, non_blocking=nb),
            padded_maps.to(DEVICE, non_blocking=nb))


def center_on_position(z, mask):
    """Subtract, per residue INDEX, the mean across the batch's proteins.

    Returns (z_centred, keep). `keep` drops indices held by fewer than
    MIN_PROTEINS_PER_INDEX proteins: such an index centres to exactly zero, and a
    zero row would enter the Barlow Twins statistics as a fake sample. In a
    length-bucketed batch chains agree on length to ~1%, so this drops very
    little -- only the tail indices of the longest chain present.

    Done in fp32 even under autocast: it is a sum over up to ~100 proteins and
    bf16 carries 8 mantissa bits.
    """
    counts = mask.sum(0)                                    # (L,) proteins per index
    zf = z.float()
    mean = (zf * mask.unsqueeze(-1)).sum(0) / counts.clamp(min=1).unsqueeze(-1)
    return zf - mean, mask & (counts >= MIN_PROTEINS_PER_INDEX)


def forward_loss(modules, batch, use_amp):
    padded_ints, mask, padded_maps = batch
    with amp.autocast(device_type=DEVICE.type, dtype=config.amp_dtype(AMP_DTYPE),
                      enabled=use_amp):
        z_seq, _ = modules["sequence_encoder"](padded_ints, mask)
        z_map, _ = modules["map_encoder"](padded_maps, mask)

        # Each branch gets its OWN mean -- they are different representations.
        # A one-protein batch has no population to average over and is left
        # alone; the residue budget makes that near-unreachable (>=3 proteins in
        # practice) but a single max-length chain could manage it.
        keep = mask
        if POSITION_CENTERING and mask.shape[0] >= MIN_PROTEINS_PER_INDEX:
            z_seq, keep = center_on_position(z_seq, mask)
            z_map, _ = center_on_position(z_map, mask)

        loss, on_diag, off_diag = modules["expanders"](z_seq, z_map, keep)
    return loss, on_diag, off_diag


def set_mode(modules, train):
    for m in modules.values():
        m.train() if train else m.eval()


# ---------------------------------------------------------------------------
# Smoke test: one train step + one val step, assert wiring before the real run
# ---------------------------------------------------------------------------
def smoke_test(modules, train_loader, val_loader, use_amp, scaler):
    """One train step + one val step, on DEVICE, at the run's REAL precision.

    use_amp/scaler are threaded in rather than hardcoded to fp32 on purpose. This
    is the check the README tells you to run before committing a GPU to a job, so
    it has to exercise the path the job will actually take: an fp32 smoke test
    cannot see a loss that only overflows under autocast, which is exactly the
    failure mode barlow_twins_core now guards against.
    """
    print(f"running smoke test (1 train step + 1 val step, "
          f"{AMP_DTYPE if use_amp else 'fp32'})...")

    # --- one train step: finite loss + grads reach ALL module groups -------
    set_mode(modules, train=True)
    for m in modules.values():
        for p in m.parameters():
            p.grad = None

    batch = to_device(next(iter(train_loader)))
    loss, on_d, off_d = forward_loss(modules, batch, use_amp)
    assert torch.isfinite(loss), (
        f"smoke: train loss is not finite ({loss.item()}) at "
        f"{AMP_DTYPE if use_amp else 'fp32'} -- on_diag {on_d.item()}, "
        f"off_diag {off_d.item()}")
    scaler.scale(loss).backward()

    for name, m in modules.items():
        assert any(p.grad is not None for p in m.parameters()), \
            f"smoke: no gradient reached {name!r} -- wiring error"

    # Clear the smoke-test grads so real training starts clean. The scaler is
    # untouched: it was only used to scale a backward we are discarding.
    for m in modules.values():
        for p in m.parameters():
            p.grad = None

    # --- one val step: finite loss under eval + no_grad --------------------
    set_mode(modules, train=False)
    with torch.no_grad():
        vbatch = to_device(next(iter(val_loader)))
        vloss, _, _ = forward_loss(modules, vbatch, use_amp)
    assert torch.isfinite(vloss), "smoke: val loss is not finite"

    print(f"smoke test passed: train loss {loss.item():.3f}, val loss {vloss.item():.3f}, "
          f"grads reached all {len(modules)} groups.\n")


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(modules, loader, optimizer, scheduler, scaler, param_list,
                    use_amp, epoch, global_step):
    set_mode(modules, train=True)
    # Running (loss, on_diag, off_diag) accumulated ON DEVICE. Calling .item() per
    # step forces a device->host sync that stalls the pipeline; these are only read
    # back at the log line and the epoch summary.
    totals = torch.zeros(3, device=DEVICE)
    n, skips = 0, 0

    for i, raw in enumerate(loader):
        batch = to_device(raw)
        optimizer.zero_grad(set_to_none=True)
        loss, on_d, off_d = forward_loss(modules, batch, use_amp)

        # NaN/inf guard: never let a bad batch poison the weights. This read does
        # sync, and it stays -- the alternative is silently training on garbage.
        if not torch.isfinite(loss):
            skips += 1
            print(f"  [epoch {epoch} step {i}] non-finite loss -> skipping optimizer step")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(param_list, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        totals += torch.stack((loss.detach(), on_d, off_d))
        n += 1
        if n % LOG_EVERY == 0:
            avg_loss, avg_on, avg_off = (totals / n).tolist()
            print(f"  [epoch {epoch} step {i}] loss {avg_loss:.3f} "
                  f"(on {avg_on:.3f}, off {avg_off:.1f}) "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    if skips:
        frac = skips / max(1, len(loader))
        warn = "  WARNING: frequent skips!" if frac > 0.05 else ""
        print(f"  epoch {epoch}: skipped {skips}/{len(loader)} batches (non-finite).{warn}")

    avg_loss, avg_on, avg_off = (totals / max(1, n)).tolist()
    return avg_loss, avg_on, avg_off, global_step


@torch.no_grad()
def validate(modules, loader, use_amp):
    set_mode(modules, train=False)
    totals = torch.zeros(3, device=DEVICE)
    n = 0
    for raw in loader:
        batch = to_device(raw)
        loss, on_d, off_d = forward_loss(modules, batch, use_amp)
        if not torch.isfinite(loss):
            continue
        totals += torch.stack((loss, on_d, off_d))
        n += 1
    avg_loss, avg_on, avg_off = (totals / max(1, n)).tolist()
    return avg_loss, avg_on, avg_off


# ---------------------------------------------------------------------------
# Free-information monitor (see the constants block for what it measures)
# ---------------------------------------------------------------------------
def build_probe(val_set):
    """A FIXED set of held-out chains, strided across the length range.

    Fixed because this metric is only readable as a trend -- resampling every
    epoch would move the number on its own. Strided rather than taken in order
    because the val loader is length-bucketed, so its first batches are all short
    chains, which would leave the length axis with nothing to condition on.

    Returns padded (ints, mask) chunks, or None when the split cannot fill the
    cells (debug runs), in which case the metric is skipped rather than quoted
    off a handful of proteins.
    """
    by_length = sorted((val_set[i][0].numel(), i) for i in range(len(val_set)))
    if len(by_length) < N_FRAC_BINS * N_LEN_BINS:
        return None

    stride = max(1, len(by_length) // PROBE_PROTEINS)
    picked = [i for _, i in by_length[::stride]][:PROBE_PROTEINS]

    chunks = []
    for s in range(0, len(picked), PROBE_CHUNK):
        seqs = [val_set[i][0] for i in picked[s:s + PROBE_CHUNK]]
        width = max(x.numel() for x in seqs)
        ints = torch.full((len(seqs), width), PAD_IDX, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for r, x in enumerate(seqs):
            ints[r, :x.numel()] = x
            mask[r, :x.numel()] = True
        chunks.append((ints.to(DEVICE), mask.to(DEVICE)))
    return chunks


def _cell_ids(mask):
    """(B, L) -> a cell index per residue, from fractional position and length."""
    B, L = mask.shape
    lengths = mask.sum(1)                                          # (B,)
    idx = torch.arange(L, device=mask.device).expand(B, L)

    frac = idx / (lengths - 1).clamp(min=1).unsqueeze(1)
    frac_bin = (frac * N_FRAC_BINS).long().clamp(0, N_FRAC_BINS - 1)

    # Log spacing: chain lengths are heavily skewed short, so linear bins would
    # drop nearly every residue into the first one.
    lo, hi = math.log(MIN_RESIDUES), math.log(config.MAX_SEQ_LENGTH)
    t = (torch.log(lengths.float().clamp(min=1)) - lo) / (hi - lo)
    len_bin = (t * N_LEN_BINS).long().clamp(0, N_LEN_BINS - 1)     # (B,)

    return frac_bin * N_LEN_BINS + len_bin.unsqueeze(1)


@torch.no_grad()
def free_information(modules, probe):
    """Fraction of z_seq's variance explained by (fractional position, length).

    A variance decomposition, not a fitted probe: the best possible predictor
    from a cell IS that cell's mean, so there is nothing to train and no
    regularisation constant to defend. Accumulated as running sums, so the
    representations never have to be held in memory at once.

    fp32 with no autocast -- a measurement should not move when --amp-dtype does.
    """
    set_mode(modules, train=False)
    encoder = modules["sequence_encoder"]
    n_cells = N_FRAC_BINS * N_LEN_BINS

    total_n, sum_sq = 0, 0.0
    sum_z = cell_sum = cell_n = None

    for ints, mask in probe:
        z = encoder(ints, mask)[0].float()
        flat = mask.reshape(-1)
        z_real = z.reshape(-1, z.shape[-1])[flat]                  # (N, D)
        cells = _cell_ids(mask).reshape(-1)[flat]                  # (N,)

        if sum_z is None:
            sum_z = torch.zeros(z_real.shape[1], device=z_real.device)
            cell_sum = torch.zeros(n_cells, z_real.shape[1], device=z_real.device)
            cell_n = torch.zeros(n_cells, device=z_real.device)

        sum_z += z_real.sum(0)
        sum_sq += z_real.pow(2).sum().item()
        total_n += z_real.shape[0]
        cell_sum.index_add_(0, cells, z_real)
        cell_n.index_add_(0, cells, torch.ones_like(cells, dtype=cell_sum.dtype))

    grand = sum_z / total_n
    grand_sq = grand.dot(grand).item()

    # var(z) = E||z||^2 - ||E z||^2, and the explained part is the same with z
    # replaced by its cell mean.
    total_var = sum_sq / total_n - grand_sq
    seen = cell_n > 0
    cell_mean = cell_sum[seen] / cell_n[seen].unsqueeze(1)
    explained = (cell_n[seen] * cell_mean.pow(2).sum(1)).sum().item() / total_n - grand_sq

    return explained / max(total_var, 1e-12)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
MODULE_KEYS = ("sequence_encoder", "map_encoder", "expanders")


def _arch_fingerprint(modules):
    """The architecture facts that decide whether a checkpoint can be reloaded.

    Stamped into every checkpoint so a later mismatch explains ITSELF instead of
    surfacing as a bare shape error. Shapes remain the authority at load time --
    this is for the diagnostic message.
    """
    map_enc = modules["map_encoder"]
    return {
        "map_seed_mode": getattr(map_enc, "seed_mode", "absolute"),
        "map_max_len": getattr(map_enc, "max_len", None),
        "param_shapes": {
            name: {k: tuple(v.shape) for k, v in modules[name].state_dict().items()}
            for name in MODULE_KEYS
        },
    }


def _split_fingerprint():
    """Everything needed to reproduce this run's train/val split.

    Stamped in because the split is otherwise only reproducible by convention:
    compare_embeddings.py has to re-derive it to say whether a protein was held
    out, and reading SEED off the checkpoint beats assuming the default was used.
    """
    return {"seed": SEED, "val_fraction": VAL_FRACTION, "min_residues": MIN_RESIDUES}


def save_checkpoint(path, epoch, modules, optimizer, scheduler, scaler, val_loss):
    # For DOWNSTREAM USE only `sequence_encoder` (and optionally `map_encoder`) are
    # needed -- the expanders are training-time scaffolding and can be dropped when
    # using the learned representations.
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "arch": _arch_fingerprint(modules),
        "split": _split_fingerprint(),
        # Which arm of the position-centering ablation produced this. Stamped
        # because the two arms are otherwise indistinguishable from the file,
        # and comparing them is the whole point of running both.
        "position_centered": POSITION_CENTERING,
        "sequence_encoder": modules["sequence_encoder"].state_dict(),
        "map_encoder": modules["map_encoder"].state_dict(),
        "expanders": modules["expanders"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def _load_module_state(module, saved, name):
    """Load `saved` into `module`, skipping tensors that no longer fit.

    Returns the list of skipped-key descriptions. Anything skipped keeps its fresh
    random init. Used instead of a strict load so that an ARCHITECTURE CHANGE costs
    you only the tensors that actually changed, rather than the whole checkpoint:
    the map encoder's seed projection was reshaped (2 -> 1000 -> 1999 columns), but
    its four transformer blocks, and all of the sequence encoder and expanders, are
    unaffected and still worth loading.
    """
    current = module.state_dict()
    compatible, skipped = {}, []
    for key, value in saved.items():
        if key not in current:
            skipped.append(f"{name}.{key}: dropped from the model")
        elif current[key].shape != value.shape:
            skipped.append(f"{name}.{key}: checkpoint {tuple(value.shape)} != "
                           f"model {tuple(current[key].shape)}")
        else:
            compatible[key] = value
    for key in current:
        if key not in saved:
            skipped.append(f"{name}.{key}: new in the model, left at init")
    module.load_state_dict(compatible, strict=False)
    return skipped


def read_checkpoint(path, map_location=None):
    """Read a checkpoint file into a dict, without touching any model.

    Split out from load_checkpoint so a caller that restores the SAME file into
    several freshly-built models pays the ~500 MB read once: `eval.py --test all`
    builds a separate model per test and would otherwise re-read it seven times.
    The result can be handed straight back to load_checkpoint().

    weights_only=False because our own checkpoints hold optimizer state
    (trusted, self-produced files).
    """
    return torch.load(path, map_location=map_location or DEVICE, weights_only=False)


def load_checkpoint(source, modules, optimizer=None, scheduler=None, scaler=None,
                    map_location=None, allow_arch_mismatch=False, label=None):
    """Restore a checkpoint saved by save_checkpoint.

    `source` is either a path or a dict already produced by read_checkpoint();
    `label` names it in messages (defaults to the path).

    Always restores the module weights, tensor by tensor, skipping any whose
    shape no longer matches the current architecture (those stay at their fresh
    init) and reporting every skip. Also restores optimizer / scheduler / scaler
    state when those objects are passed in.

    RESUMING across an architecture change is refused. AdamW's saved moments are
    shape-bound to the parameters that produced them, so loading them onto a
    reshaped parameter either raises deep inside the first step() or silently
    corrupts the update. Pass allow_arch_mismatch=True to warm-start from the
    compatible weights instead -- optimizer/scheduler/scaler state is then dropped
    and the run restarts its LR schedule from epoch 1.

    Returns (epoch, val_loss, skipped) -- skipped is the list of tensors that could
    not be carried over, so callers can flag results computed on partly-random
    weights.
    """
    ckpt = source if isinstance(source, dict) else read_checkpoint(source, map_location)
    path = label or (source if isinstance(source, str) else "<checkpoint>")

    skipped = []
    for name in MODULE_KEYS:
        if name not in ckpt:
            skipped.append(f"{name}.*: absent from the checkpoint, left at init")
            continue
        skipped += _load_module_state(modules[name], ckpt[name], name)

    # Whole modules the checkpoint carries that this architecture no longer has --
    # e.g. `predictor`, deleted when the prediction head was removed. Dropping
    # these is correct, but it is still an architecture mismatch and must be
    # reported as one, not silently ignored.
    for name in ckpt:
        if name in MODULE_KEYS or name in ("epoch", "val_loss", "arch", "split",
                                           "optimizer", "scheduler", "scaler"):
            continue
        skipped.append(f"{name}.*: module no longer exists, discarded")

    resuming = optimizer is not None or scheduler is not None or scaler is not None

    if skipped:
        saved_arch = ckpt.get("arch", {})
        print(f"WARNING: '{path}' was written by a DIFFERENT architecture. "
              f"{len(skipped)} tensor(s) could not be carried over and are at their "
              f"fresh random init:")
        for line in skipped:
            print(f"    {line}")
        print(f"    checkpoint map-encoder seeding: "
              f"{saved_arch.get('map_seed_mode', 'unknown (pre-dates the arch stamp)')}"
              f" / max_len {saved_arch.get('map_max_len', 'unknown')}")
        print(f"    this model's                  : "
              f"{getattr(modules['map_encoder'], 'seed_mode', '?')} / max_len "
              f"{getattr(modules['map_encoder'], 'max_len', '?')}")

        if resuming and not allow_arch_mismatch:
            raise RuntimeError(
                f"cannot RESUME training from '{path}': the architecture changed, so "
                f"the saved optimizer moments no longer match {len(skipped)} "
                f"parameter(s) (see the list above). Either train from scratch, or "
                f"pass allow_arch_mismatch=True to warm-start from the compatible "
                f"weights with a fresh optimizer and LR schedule."
            )
        print("    -> any measurement that reads a module listed above is being "
              "computed on partly-random weights.")

    if skipped:
        # Reached only with allow_arch_mismatch=True (the strict path raised above).
        if resuming:
            print("    -> allow_arch_mismatch: optimizer/scheduler/scaler state "
                  "dropped; this is a WARM START, not a resume. Training restarts "
                  "at epoch 1 with a fresh LR schedule.")
        return 0, float("inf"), skipped

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf")), skipped


def load_sequence_encoder(path=None, device=None):
    """DOWNSTREAM USE: rebuild just the trained sequence encoder and return it in
    eval() mode. The map encoder and expanders are training-time scaffolding and
    are ignored here -- the sequence encoder is what produces the
    representations you actually use.

        enc = load_sequence_encoder()
        seq_repr, mask = enc(padded_ints, mask)   # (B, L, 512) per-residue reps

    path/device default to CKPT_DIR/best.pt on DEVICE, resolved at CALL time so
    they honour --ckpt-dir / $CKPT_DIR (a default argument would freeze the value
    at import time, before any override ran).
    """
    path = path or os.path.join(CKPT_DIR, "best.pt")
    device = device or DEVICE
    ckpt = read_checkpoint(path, map_location=device)
    enc = SequenceEncoder().to(device)
    skipped = _load_module_state(enc, ckpt["sequence_encoder"], "sequence_encoder")
    if skipped:
        # The map encoder's seeding has been reshaped more than once; the sequence
        # encoder has not. A mismatch HERE means the branch you are about to use
        # downstream is partly random, which must not pass silently.
        raise RuntimeError(
            f"'{path}' does not match the current SequenceEncoder:\n    "
            + "\n    ".join(skipped)
        )
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(smoke_only=False):
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_loader, val_loader, n_train, n_val = build_loaders()
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch

    modules = build_modules()
    optimizer, scheduler, param_list = build_optimizer(modules, total_steps)

    # AMP is a CUDA-only win here; --no-amp forces full fp32 even on a GPU.
    use_amp = USE_AMP and DEVICE.type == "cuda"
    # bf16 carries fp32's exponent range, so loss scaling buys nothing -- the
    # scaler is enabled for fp16 only. Disabled, every scaler call is a no-op.
    scaler = amp.GradScaler(device=DEVICE.type,
                            enabled=use_amp and AMP_DTYPE == "fp16")

    # --- config summary ----------------------------------------------------
    n_params = {name: sum(p.numel() for p in m.parameters()) for name, m in modules.items()}
    print("=" * 64)
    print("Protein Barlow - joint training")
    print(f"  device            : {DEVICE}" +
          (f" ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else ""))
    print(f"  data / ckpt dir   : {DATA_DIR} -> {CKPT_DIR}")
    print(f"  train / val prot. : {n_train} / {n_val}   (val_fraction {VAL_FRACTION}, "
          f"min length {MIN_RESIDUES})")
    # Batches are length-bucketed under a residue budget, so proteins-per-batch
    # varies (many short chains, few long ones). Report what that works out to.
    batch_sizes = train_loader.batch_sampler.batch_sizes
    print(f"  batching          : <={RESIDUES_PER_BATCH} residues/batch -> "
          f"{min(batch_sizes)}-{max(batch_sizes)} proteins (median "
          f"{int(np.median(batch_sizes))}), {steps_per_epoch} steps/epoch, "
          f"workers {NUM_WORKERS}")
    print(f"  epochs            : {EPOCHS}   total steps {total_steps}")
    print(f"  lr / weight_decay : {LR} / {WEIGHT_DECAY}   warmup {WARMUP_STEPS}")
    print(f"  position centring : {'ON' if POSITION_CENTERING else 'OFF (ablation B)'}")
    print(f"  grad clip / amp   : {GRAD_CLIP} / "
          f"{AMP_DTYPE if use_amp else 'off (fp32)'}"
          f"{'' if not use_amp else ' (scaler ' + ('on' if scaler.is_enabled() else 'off') + ')'}")
    print(f"  params (M)        : " +
          ", ".join(f"{k} {v/1e6:.1f}" for k, v in n_params.items()) +
          f"  | total {sum(n_params.values())/1e6:.1f}M")
    print("=" * 64)

    # --- optional resume ---------------------------------------------------
    best_val = float("inf")
    start_epoch = 1
    if RESUME_FROM is not None:
        last_epoch, best_val, skipped = load_checkpoint(
            RESUME_FROM, modules, optimizer, scheduler, scaler,
            allow_arch_mismatch=WARM_START,
        )
        start_epoch = last_epoch + 1
        # Batch composition is a function of (seed, epoch); without this a resumed
        # run would replay epoch 1's batches.
        train_loader.batch_sampler.set_epoch(start_epoch - 1)
        if skipped:
            # Warm start: load_checkpoint dropped the optimizer state and reset the
            # epoch, so the LR schedule and best-val tracking start clean.
            print(f"warm-started from {RESUME_FROM}: {len(skipped)} tensor(s) "
                  f"reinitialised, training from epoch 1 with a fresh schedule.")
        else:
            print(f"resumed from {RESUME_FROM}: continuing at epoch {start_epoch} "
                  f"(best val so far {best_val:.3f})")

    # --- catch wiring errors in seconds ------------------------------------
    smoke_test(modules, train_loader, val_loader, use_amp, scaler)
    if smoke_only:
        print("--smoke-test: wiring check only, exiting before training.")
        return

    probe = build_probe(val_loader.dataset)
    if probe is None:
        print("free-information monitor: OFF (val split too small to fill the cells)")

    # --- real run ----------------------------------------------------------
    global_step = (start_epoch - 1) * steps_per_epoch
    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loss, tr_on, tr_off, global_step = train_one_epoch(
            modules, train_loader, optimizer, scheduler, scaler, param_list,
            use_amp, epoch, global_step,
        )
        va_loss, va_on, va_off = validate(modules, val_loader, use_amp)

        # Logged EVERY epoch, not spot-checked: the shortcut is an attractor the
        # model can drift into at any point, and val loss falls as it drifts.
        # A climbing `free` is the signal to kill the run -- note best.pt below
        # is still selected on val loss, which cannot see this.
        free = free_information(modules, probe) if probe else None

        print(f"epoch {epoch:3d} | train {tr_loss:.3f} (on {tr_on:.3f}, off {tr_off:.1f}) "
              f"| val {va_loss:.3f} (on {va_on:.3f}, off {va_off:.1f})"
              + (f" | free {free:.3f}" if free is not None else ""))

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            # NOTE: in debug mode CKPT_PREFIX is "debug_", so these throwaway
            # checkpoints never overwrite real ones. Saving is otherwise untouched.
            save_checkpoint(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}best.pt"),
                            epoch, modules, optimizer, scheduler, scaler, va_loss)
            print(f"  new best val {best_val:.3f} -> saved {CKPT_PREFIX}best.pt")

        if epoch % CKPT_EVERY_EPOCHS == 0:
            # Rolling by default: each of these is ~500 MB (weights + optimizer
            # moments), and only the newest is a useful crash-resume point.
            # --keep-epoch-ckpts keeps the whole history instead.
            name = (f"{CKPT_PREFIX}epoch_{epoch:03d}.pt" if KEEP_EPOCH_CKPTS
                    else f"{CKPT_PREFIX}last.pt")
            path = os.path.join(CKPT_DIR, name)
            save_checkpoint(path, epoch, modules, optimizer, scheduler, scaler, va_loss)
            print(f"  saved {path}")

    print(f"done. best val loss {best_val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Barlow joint training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_override_args(parser)

    hp = parser.add_argument_group("hyperparameters")
    hp.add_argument("--epochs", type=int, default=None)
    hp.add_argument("--lr", type=float, default=None)
    hp.add_argument("--weight-decay", type=float, default=None)
    hp.add_argument("--warmup-steps", type=int, default=None)
    hp.add_argument("--seed", type=int, default=None)

    rn = parser.add_argument_group("run control")
    rn.add_argument("--resume", default=None,
                    help="checkpoint path to continue from (fresh run if omitted)")
    rn.add_argument("--warm-start", action="store_true", dest="warm_start",
                    help="allow --resume across an architecture change: keep the "
                         "weights that still fit, reinitialise the rest, drop the "
                         "optimizer state and restart the LR schedule")
    rn.add_argument("--ckpt-every", type=int, default=None,
                    help="also save every N epochs, as a rolling last.pt "
                         "(best.pt is always saved)")
    rn.add_argument("--keep-epoch-ckpts", action="store_true", dest="keep_epoch_ckpts",
                    help="write epoch_NNN.pt every --ckpt-every epochs instead of "
                         "overwriting one last.pt (~500 MB each)")
    rn.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision and train in full fp32")
    rn.add_argument("--no-position-centering", action="store_true",
                    dest="no_position_centering",
                    help="do NOT subtract the per-index mean before the loss. "
                         "This is the B arm of the ablation -- it restores the "
                         "positional shortcut, so expect `free` to climb")
    rn.add_argument("--debug", action="store_true",
                    help="fast GPU-free pass over a tiny subset (cpu, 20 proteins, "
                         "2 epochs, batch 4) to catch shape/mask/wiring bugs")
    rn.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                    help="run only the wiring check on DEVICE, then exit")
    args = parser.parse_args()

    # Overrides must be applied BEFORE main() builds the data/model/optimizer.
    # With no flags at all, nothing below changes and the run is exactly as before.
    # --debug is applied LAST so its tiny-subset settings win over everything.
    apply_cli_overrides(args)
    if args.debug:
        apply_debug_overrides()
    main(smoke_only=args.smoke_test)
