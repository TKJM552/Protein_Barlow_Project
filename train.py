"""Joint training loop for the protein JEPA.

Wires together the already-built components (imported, not reimplemented):

  DataLoader -> SequenceEncoder -> Predictor -> pred  \
                                                       Barlow Twins loss
                MapEncoder ------------------------> target /

This is JOINT training: ONE AdamW optimizer holds the parameters of the sequence
encoder, the predictor, the map encoder, AND the Barlow Twins expanders. There is
no frozen branch, no stop-gradient, and no EMA -- collapse is prevented purely by
the loss's off-diagonal term (see barlow_twins.py).
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
    collate_pad,
    BATCH_SIZE,
)
from predictor import build_predictor
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

LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 50
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
VAL_FRACTION = 0.1
SEED = 0
CKPT_EVERY_EPOCHS = 5
LOG_EVERY = 25              # print a running train-loss line every N steps
RESUME_FROM = None         # --resume PATH to continue a run; None = fresh run
USE_AMP = True             # --no-amp to force full fp32 (cpu runs ignore this)

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
    global DEVICE, USE_AMP

    # argparse dest -> the module-level constant it overrides.
    for dest, constant in [
        ("data_dir", "DATA_DIR"), ("ckpt_dir", "CKPT_DIR"),
        ("num_workers", "NUM_WORKERS"), ("amp_dtype", "AMP_DTYPE"),
        ("batch_size", "BATCH_SIZE"), ("epochs", "EPOCHS"), ("lr", "LR"),
        ("weight_decay", "WEIGHT_DECAY"), ("warmup_steps", "WARMUP_STEPS"),
        ("seed", "SEED"), ("ckpt_every", "CKPT_EVERY_EPOCHS"),
        ("resume", "RESUME_FROM"),
    ]:
        value = getattr(args, dest, None)
        if value is not None:
            globals()[constant] = value

    # These two need more than a straight assignment.
    if getattr(args, "device", None) is not None:
        DEVICE = config.resolve_device(args.device)
    if getattr(args, "no_amp", False):
        USE_AMP = False

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
                   help="proteins per batch (env: BATCH_SIZE)")
    return g


def apply_debug_overrides():
    """Fast, GPU-free bug-catching pass over a tiny subset.

    Called ONLY when --debug is passed, and only AFTER the constants above are
    defined -- these four overrides plus the smaller dataset are the *only*
    differences from a real run. Every other code path (model construction,
    optimizer, loss, training loop, collate) is identical, so shape/mask/wiring
    bugs surface here exactly as they would on a GPU.
    """
    global DEVICE, EPOCHS, BATCH_SIZE, DEBUG_MAX_PROTEINS, CKPT_PREFIX
    DEVICE = torch.device("cpu")   # torch.device (not the str "cpu") so DEVICE.type
                                   # still works for autocast / GradScaler / .to()
    EPOCHS = 2
    BATCH_SIZE = 4                 # several small batches -> exercises padding/masking
    DEBUG_MAX_PROTEINS = 20
    CKPT_PREFIX = "debug_"
    print("*** DEBUG MODE: cpu, 20 proteins, 2 epochs, batch 4 -- bug-catching only, "
          "not a real run ***")


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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            **loader_kwargs)
    return train_loader, val_loader, n_train, n_val


# ---------------------------------------------------------------------------
# Modules + optimizer + scheduler
# ---------------------------------------------------------------------------
def build_modules():
    return {
        "sequence_encoder": SequenceEncoder().to(DEVICE),
        "predictor": build_predictor().to(DEVICE),
        "map_encoder": ContactMapEncoder().to(DEVICE),
        "expanders": BarlowTwinsLoss().to(DEVICE),   # the loss module owns the expanders
    }


def build_optimizer(modules, total_steps):
    # ONE optimizer over ALL parameters: encoder + predictor + map encoder + expanders.
    param_list = []
    for name, m in modules.items():
        params = list(m.parameters())
        assert len(params) > 0, f"module {name!r} contributed zero parameters"
        param_list += params
    assert len(param_list) > 0, "optimizer received no parameters"

    optimizer = torch.optim.AdamW(param_list, lr=LR, weight_decay=WEIGHT_DECAY)

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


def forward_loss(modules, batch, use_amp):
    padded_ints, mask, padded_maps = batch
    with amp.autocast(device_type=DEVICE.type, dtype=config.amp_dtype(AMP_DTYPE),
                      enabled=use_amp):
        seq_repr, _ = modules["sequence_encoder"](padded_ints, mask)
        pred, _ = modules["predictor"](seq_repr, mask)
        target, _ = modules["map_encoder"](padded_maps, mask)
        loss, on_diag, off_diag = modules["expanders"](pred, target, mask)
    return loss, on_diag, off_diag


def set_mode(modules, train):
    for m in modules.values():
        m.train() if train else m.eval()


# ---------------------------------------------------------------------------
# Smoke test: one train step + one val step, assert wiring before the real run
# ---------------------------------------------------------------------------
def smoke_test(modules, train_loader, val_loader):
    print("running smoke test (1 train step + 1 val step)...")

    # --- one train step: finite loss + grads reach ALL four groups ---------
    set_mode(modules, train=True)
    for m in modules.values():
        for p in m.parameters():
            p.grad = None

    batch = to_device(next(iter(train_loader)))
    loss, on_d, off_d = forward_loss(modules, batch, use_amp=False)
    assert torch.isfinite(loss), "smoke: train loss is not finite"
    loss.backward()

    for name, m in modules.items():
        assert any(p.grad is not None for p in m.parameters()), \
            f"smoke: no gradient reached {name!r} -- wiring error"

    # Clear the smoke-test grads so real training starts clean.
    for m in modules.values():
        for p in m.parameters():
            p.grad = None

    # --- one val step: finite loss under eval + no_grad --------------------
    set_mode(modules, train=False)
    with torch.no_grad():
        vbatch = to_device(next(iter(val_loader)))
        vloss, _, _ = forward_loss(modules, vbatch, use_amp=False)
    assert torch.isfinite(vloss), "smoke: val loss is not finite"

    print(f"smoke test passed: train loss {loss.item():.3f}, val loss {vloss.item():.3f}, "
          f"grads reached all 4 groups.\n")


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(modules, loader, optimizer, scheduler, scaler, param_list,
                    use_amp, epoch, global_step):
    set_mode(modules, train=True)
    running, running_on, running_off, n = 0.0, 0.0, 0.0, 0
    skips = 0

    for i, raw in enumerate(loader):
        batch = to_device(raw)
        optimizer.zero_grad(set_to_none=True)
        loss, on_d, off_d = forward_loss(modules, batch, use_amp)

        # NaN/inf guard: never let a bad batch poison the weights.
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

        running += loss.item()
        running_on += on_d.item()
        running_off += off_d.item()
        n += 1
        if n % LOG_EVERY == 0:
            print(f"  [epoch {epoch} step {i}] loss {running / n:.3f} "
                  f"(on {running_on / n:.3f}, off {running_off / n:.1f}) "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    if skips:
        frac = skips / max(1, len(loader))
        warn = "  WARNING: frequent skips!" if frac > 0.05 else ""
        print(f"  epoch {epoch}: skipped {skips}/{len(loader)} batches (non-finite).{warn}")

    avg = running / max(1, n)
    return avg, running_on / max(1, n), running_off / max(1, n), global_step


@torch.no_grad()
def validate(modules, loader, use_amp):
    set_mode(modules, train=False)
    total, total_on, total_off, n = 0.0, 0.0, 0.0, 0
    for raw in loader:
        batch = to_device(raw)
        loss, on_d, off_d = forward_loss(modules, batch, use_amp)
        if not torch.isfinite(loss):
            continue
        total += loss.item()
        total_on += on_d.item()
        total_off += off_d.item()
        n += 1
    return total / max(1, n), total_on / max(1, n), total_off / max(1, n)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path, epoch, modules, optimizer, scheduler, scaler, val_loss):
    # For DOWNSTREAM USE only `sequence_encoder` (and optionally `map_encoder`) are
    # needed -- the predictor and the expanders are training-time scaffolding and
    # can be dropped when using the learned representations.
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "sequence_encoder": modules["sequence_encoder"].state_dict(),
        "map_encoder": modules["map_encoder"].state_dict(),
        "predictor": modules["predictor"].state_dict(),
        "expanders": modules["expanders"].state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def load_checkpoint(path, modules, optimizer=None, scheduler=None, scaler=None,
                    map_location=None):
    """RESUME a training run from a checkpoint saved by save_checkpoint.

    Always restores the four module weights. Also restores optimizer / scheduler /
    scaler state when those objects are passed in. Returns (epoch, val_loss) so the
    caller can continue from epoch + 1. weights_only=False because our own
    checkpoints hold optimizer/scheduler state (trusted, self-produced files).
    """
    ckpt = torch.load(path, map_location=map_location or DEVICE, weights_only=False)
    modules["sequence_encoder"].load_state_dict(ckpt["sequence_encoder"])
    modules["map_encoder"].load_state_dict(ckpt["map_encoder"])
    modules["predictor"].load_state_dict(ckpt["predictor"])
    modules["expanders"].load_state_dict(ckpt["expanders"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def load_sequence_encoder(path=None, device=None):
    """DOWNSTREAM USE: rebuild just the trained sequence encoder and return it in
    eval() mode. The predictor, map encoder, and expanders are training-time
    scaffolding and are ignored here -- the sequence encoder is what produces the
    representations you actually use.

        enc = load_sequence_encoder()
        seq_repr, mask = enc(padded_ints, mask)   # (B, L, 512) per-residue reps

    path/device default to CKPT_DIR/best.pt on DEVICE, resolved at CALL time so
    they honour --ckpt-dir / $CKPT_DIR (a default argument would freeze the value
    at import time, before any override ran).
    """
    path = path or os.path.join(CKPT_DIR, "best.pt")
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc = SequenceEncoder().to(device)
    enc.load_state_dict(ckpt["sequence_encoder"])
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
    print("Protein JEPA - joint training")
    print(f"  device            : {DEVICE}" +
          (f" ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else ""))
    print(f"  data / ckpt dir   : {DATA_DIR} -> {CKPT_DIR}")
    print(f"  train / val prot. : {n_train} / {n_val}   (val_fraction {VAL_FRACTION})")
    print(f"  batch size        : {BATCH_SIZE}   steps/epoch {steps_per_epoch}   "
          f"workers {NUM_WORKERS}")
    print(f"  epochs            : {EPOCHS}   total steps {total_steps}")
    print(f"  lr / weight_decay : {LR} / {WEIGHT_DECAY}   warmup {WARMUP_STEPS}")
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
        last_epoch, best_val = load_checkpoint(RESUME_FROM, modules, optimizer,
                                               scheduler, scaler)
        start_epoch = last_epoch + 1
        print(f"resumed from {RESUME_FROM}: continuing at epoch {start_epoch} "
              f"(best val so far {best_val:.3f})")

    # --- catch wiring errors in seconds ------------------------------------
    smoke_test(modules, train_loader, val_loader)
    if smoke_only:
        print("--smoke-test: wiring check only, exiting before training.")
        return

    # --- real run ----------------------------------------------------------
    global_step = (start_epoch - 1) * steps_per_epoch
    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loss, tr_on, tr_off, global_step = train_one_epoch(
            modules, train_loader, optimizer, scheduler, scaler, param_list,
            use_amp, epoch, global_step,
        )
        va_loss, va_on, va_off = validate(modules, val_loader, use_amp)

        print(f"epoch {epoch:3d} | train {tr_loss:.3f} (on {tr_on:.3f}, off {tr_off:.1f}) "
              f"| val {va_loss:.3f} (on {va_on:.3f}, off {va_off:.1f})")

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            # NOTE: in debug mode CKPT_PREFIX is "debug_", so these throwaway
            # checkpoints never overwrite real ones. Saving is otherwise untouched.
            save_checkpoint(os.path.join(CKPT_DIR, f"{CKPT_PREFIX}best.pt"),
                            epoch, modules, optimizer, scheduler, scaler, va_loss)
            print(f"  new best val {best_val:.3f} -> saved {CKPT_PREFIX}best.pt")

        if epoch % CKPT_EVERY_EPOCHS == 0:
            path = os.path.join(CKPT_DIR, f"{CKPT_PREFIX}epoch_{epoch:03d}.pt")
            save_checkpoint(path, epoch, modules, optimizer, scheduler, scaler, va_loss)
            print(f"  saved {path}")

    print(f"done. best val loss {best_val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein JEPA joint training",
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
    rn.add_argument("--ckpt-every", type=int, default=None,
                    help="also save every N epochs (best.pt is always saved)")
    rn.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision and train in full fp32")
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
