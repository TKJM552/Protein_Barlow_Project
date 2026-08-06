"""Train the contact predictor, and answer whether pretraining was worth anything.

This script exists to run ONE experiment in two arms:

    --arm pretrained   frozen z_seq from a Barlow Twins checkpoint  -> ContactPredictor
    --arm scratch      nn.Embedding(21, 512) trained from scratch   -> ContactPredictor

Everything else is identical: same split, same seed, same architecture, same
budget. **The delta between the arms is the result.** Nothing else in this repo
establishes that the pretrained encoder is worth more than feeding raw amino
acids, and FINDINGS lists this as the highest-value open item precisely because
every diagnostic so far has measured a proxy instead.

Read the numbers in this order:

  1. Does EITHER arm beat DISTANCE-ONLY on long-range P@L/5? If not, nothing that
     follows means anything -- the model learned the |i-j| prior and stopped.
     FINDINGS has distance-only at AUC 0.758, ahead of the pretrained encoder's
     0.636, so this is a live risk rather than a formality.
  2. Does `pretrained` beat `scratch`? That is the pretraining question.
  3. Only then, the absolute numbers.

Usage:
    python train_contact.py --arm scratch
    python train_contact.py --arm pretrained --encoder-ckpt /workspace/checkpoints/best.pt
    python train_contact.py --arm scratch --smoke-test          # wiring only
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, Sampler

import config
import train
from seq_encoder import (EMBED_DIM, VOCAB_SIZE, PAD_IDX, collate_pad,
                         dataset_lengths)
from contact_predictor import (
    ContactPredictor, contact_loss, crop_pair, pair_mask_for,
    separation_weight, CROP,
)

# --- training ---------------------------------------------------------------
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
GRAD_CLIP = 1.0
LOG_EVERY = 50

# Batches are sized by PAIR count, not residue count -- see PairBudgetSampler.
# 1.5M cells is ~11 GB of stored activations at PAIR_DIM=64 with checkpointing on,
# by the estimate in contact_predictor's docstring. That estimate is derived from
# the code rather than measured, so the first epoch prints the REAL peak and this
# should be retuned from that number, not from the arithmetic.
PAIR_BUDGET = 1_500_000

# --- evaluation -------------------------------------------------------------
MIN_SEP = 12          # this repo's "long-range". NOTE the contact-prediction
                      # literature usually means |i-j| >= 24 and calls 12..23
                      # medium-range, so these numbers sit on an easier bar than
                      # a published one. Both are reported; compare like with like.
STRICT_SEP = 24
EVAL_PROTEINS = 200   # fixed set, sampled once, so the epoch-to-epoch curve moves
                      # because the MODEL moved
MAX_EVAL_LEN = 990    # whole proteins, uncropped, one at a time. The default
                      # admits the longest chain in the set; lower it only if
                      # eval OOMs, and note that P@L/5 depends on L so the pool's
                      # length distribution is part of the number.
WEIGHT_EST_BATCHES = 30


# ---------------------------------------------------------------------------
# Batching: pairs, not residues
# ---------------------------------------------------------------------------
class PairBudgetSampler(Sampler):
    """Group proteins so that padded PAIR count (B * L_max^2) stays under budget.

    train.py budgets RESIDUES (B * L_max <= 4096), which is right for the encoder,
    whose cost is linear in length. It bounds nothing here. Both of these are 4096
    residues:

        4 proteins x 1024 residues  ->  4 * 1024^2 = 4,194,304 pairs
       20 proteins x  205 residues  ->  20 * 205^2 =   840,000 pairs

    Five times the memory from batches that look identical by the residue count.
    Budgeting pairs directly makes every batch cost about the same: long proteins
    land in small batches, short ones in large batches, and no crop is needed.

    Sorting by length before grouping keeps padding low, the same reason
    LengthBucketSampler does it. Batch ORDER is reshuffled each epoch so the model
    does not see lengths in a fixed sequence; the contents of a batch are fixed,
    which is what makes the memory bound hold.

    **The budget has a FLOOR, and it is L_max^2.** A protein larger than the whole
    budget is still emitted alone -- dropping it would silently bias training
    toward short chains -- so the longest chain in the set costs 990^2 = 980,100
    pairs no matter how low --pair-budget goes. Lowering the budget past that point
    buys nothing; only --crop does, at the cost of never training on pairs further
    apart than the crop. On a 24 GB card this is a non-issue (~7.5 GB checkpointed),
    but it is why a CPU run needs --crop to finish at all.
    """

    def __init__(self, lengths, budget=PAIR_BUDGET, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.budget = budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._build()

    def _build(self):
        order = np.argsort(self.lengths, kind="stable")
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = int(self.lengths[i])
            new_max = max(cur_max, L)
            # A protein bigger than the whole budget still gets emitted alone --
            # dropping it would silently bias the training set toward short chains.
            if cur and (len(cur) + 1) * new_max * new_max > self.budget:
                batches.append(cur)
                cur, cur_max = [int(i)], L
            else:
                cur.append(int(i))
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        for k in order:
            yield self.batches[k]

    def __len__(self):
        return len(self.batches)


def build_pair_loaders(seed, budget):
    """Train/val loaders over train.py's OWN cluster-grouped split.

    build_loaders() is called for the split, not the loaders: a random split puts
    40.3% of val chains' exact sequence twins in train (measured, FINDINGS) and
    both arms would look excellent. Its residue-budgeted loaders are then replaced
    with pair-budgeted ones.
    """
    tr0, va0, n_train, n_val = train.build_loaders()
    train_set, val_set = tr0.dataset, va0.dataset
    kw = dict(collate_fn=collate_pad, num_workers=train.NUM_WORKERS,
              pin_memory=train.DEVICE.type == "cuda")
    tr = DataLoader(train_set, batch_sampler=PairBudgetSampler(
        dataset_lengths(train_set), budget, shuffle=True, seed=seed), **kw)
    return tr, val_set, n_train, n_val


# ---------------------------------------------------------------------------
# The only thing that differs between the two arms
# ---------------------------------------------------------------------------
class PretrainedInput(nn.Module):
    """Pretrained sequence encoder. (B,L) ints -> (B,L,512).

    FROZEN by default, and that is the scientific setting: the question is what the
    pretrained representation ALREADY contains. Once the encoder can adapt, a win no
    longer distinguishes "the representation was good" from "18.9M extra trainable
    parameters were good", and the scratch arm has no matching 18.9M to offer.

    --unfreeze is the PRODUCT setting. Fine-tuning normally beats frozen features,
    because z_seq was optimised to make two views agree, not to predict contacts,
    and fine-tuning lets nearly-right features move toward what this task needs.

    Three things it costs, all deliberate:

      * trainable parameters go 7.2M -> 26.1M, and memorisation is this repo's
        measured failure (a 48.6x train/val gap that more data did not fix)
      * the JOINT embedding is not preserved. Nothing holds z_seq in the shared
        space with z_map any more -- no map encoder, no Barlow Twins term. The
        result is a supervised contact predictor that started from a joint
        embedding, not a joint embedding. (best.pt on disk is untouched; this
        writes a new model.)
      * attribution is lost, per the first paragraph.

    Worth running BOTH, because the gap between them measures the representation:
    frozen ~= fine-tuned means the pretraining already held what contacts need;
    fine-tuned >> frozen means the encoder had to change substantially to be useful.
    """

    def __init__(self, ckpt_path, device, unfreeze=False, random_init=False, seed=0):
        super().__init__()
        if random_init:
            # The DEPTH control. Identical architecture, identical freezing, only
            # the weights differ -- so `pretrained` minus `random` isolates what
            # pretraining contributed, with no confound.
            #
            # It is needed because `pretrained` vs `scratch` compares 8 transformer
            # blocks against 2: six frozen encoder blocks plus the predictor's two,
            # versus the predictor's two alone. A win there could be depth rather
            # than pretraining, and this arm is what separates them.
            #
            # Offset seed so these weights are not the predictor's init draw. Same
            # convention as eval.py's random-encoder baselines.
            train.set_seed(seed + 999)
            self.encoder = train.SequenceEncoder().to(device)
        else:
            self.encoder = train.load_sequence_encoder(ckpt_path, device)
        self.random_init = random_init
        self.unfrozen = unfreeze
        for p in self.encoder.parameters():
            p.requires_grad_(unfreeze)
        if not unfreeze:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.unfrozen:
            # Stay in eval() whatever the training loop does, so the frozen
            # encoder's dropout never fires and z_seq is deterministic per chain.
            self.encoder.eval()
        return self

    def forward(self, ints, mask):
        if self.unfrozen:
            return self.encoder(ints, mask)[0]
        with torch.no_grad():
            return self.encoder(ints, mask)[0].detach()


class ScratchInput(nn.Module):
    """A plain embedding lookup, trained with the predictor. The honest control.

    21 x 512 = 10,752 trainable parameters against the pretrained arm's 18.9M
    frozen ones. That asymmetry IS the experiment, not a flaw in it: if the tiny
    lookup matches the pretrained encoder, the pretraining bought nothing.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)

    def forward(self, ints, mask):
        return self.embed(ints)


def build_input(arm, ckpt_path, device, unfreeze=False, seed=0):
    if arm == "pretrained":
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise SystemExit(
                f"--arm pretrained needs --encoder-ckpt pointing at a Barlow Twins "
                f"checkpoint; got {ckpt_path!r}")
        return PretrainedInput(ckpt_path, device, unfreeze).to(device)
    if unfreeze:
        raise SystemExit(f"--unfreeze applies only to --arm pretrained. The scratch "
                         f"arm's embedding is trained either way, and the random arm "
                         f"is frozen by definition -- unfreezing it would just be a "
                         f"bigger scratch arm.")
    if arm == "random":
        return PretrainedInput(None, device, unfreeze=False,
                               random_init=True, seed=seed).to(device)
    return ScratchInput().to(device)


def param_groups(model, provider, lr, encoder_lr):
    """Two learning rates: the pretrained encoder gets the smaller one.

    A single LR across both goes badly. Early on the predictor is random and
    producing nonsense, so large updates flow into pretrained weights before there
    is anything useful to steer them with -- the encoder drifts away from what
    pretraining gave it before the head can exploit it (catastrophic forgetting).
    A ~10x smaller LR on pretrained weights is the standard mitigation.
    """
    head = [p for p in model.parameters() if p.requires_grad]
    enc = [p for p in provider.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": lr}]
    if enc:
        # ScratchInput's embedding is NEW, not pretrained, so it belongs with the
        # head at the full rate. Only a fine-tuned encoder gets the reduced one.
        is_pretrained = isinstance(provider, PretrainedInput)
        groups.append({"params": enc, "lr": encoder_lr if is_pretrained else lr})
    return groups


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_l5(scores, cmap, min_sep):
    """Top-L/5 precision over UPPER-TRIANGLE pairs with j - i > min_sep.

    Upper triangle only: the map is symmetric, so counting both (i,j) and (j,i)
    would score every prediction twice and make L/5 mean half as much.

    Returns None when a protein has fewer than L/5 eligible pairs, so short chains
    are skipped rather than contributing a degenerate 0 or 1.
    """
    L = cmap.shape[0]
    idx = torch.arange(L, device=cmap.device)
    keep = (idx[None, :] - idx[:, None]) > min_sep
    s, t = scores[keep], cmap[keep]
    k = max(1, L // 5)
    if s.numel() < k:
        return None
    return t[s.topk(k).indices].mean().item()


def distance_only_scores(L, device):
    """The trivial baseline: nearer in the chain scores higher. Reads no chemistry.

    In every eval because FINDINGS measured it BEATING the pretrained encoder
    (AUC 0.758 vs 0.636). A contact model that does not clear it has learned the
    offset prior and nothing else.
    """
    idx = torch.arange(L, device=device)
    return -(idx[:, None] - idx[None, :]).abs().float()


@torch.no_grad()
def evaluate(model, provider, dataset, indices, device, pos_w, long_w,
             amp_dtype=None, max_len=MAX_EVAL_LEN):
    """Full-length, uncropped, one protein at a time. Returns a dict of means."""
    model.eval()
    provider.eval()
    losses, rows = [], {MIN_SEP: ([], []), STRICT_SEP: ([], [])}

    for i in indices:
        seq, cmap = dataset[i]
        L = seq.shape[0]
        if L > max_len:
            continue
        ints = seq.unsqueeze(0).to(device)
        maps = cmap.unsqueeze(0).to(device)
        mask = torch.ones(1, L, dtype=torch.bool, device=device)

        with amp.autocast(device_type=device.type, dtype=amp_dtype,
                          enabled=amp_dtype is not None):
            logits = model(provider(ints, mask), mask)
        logits = logits.float()

        losses.append(contact_loss(logits, maps, mask, pos_w,
                                   separation_weight(mask, MIN_SEP, long_w)).item())
        dist = distance_only_scores(L, device)
        for sep, (mine, theirs) in rows.items():
            pm = precision_at_l5(logits[0], maps[0], sep)
            if pm is not None:
                mine.append(pm)
                theirs.append(precision_at_l5(dist, maps[0], sep))

    model.train()
    provider.train()
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return {"loss": mean(losses), "n": len(rows[MIN_SEP][0]),
            "p": mean(rows[MIN_SEP][0]), "p_dist": mean(rows[MIN_SEP][1]),
            "p_strict": mean(rows[STRICT_SEP][0]),
            "p_strict_dist": mean(rows[STRICT_SEP][1])}


def save_checkpoint(path, args, model, provider, epoch, metrics,
                    opt=None, sched=None, scaler=None, best=None):
    """Everything needed to reproduce the number, not just the weights.

    The two arms differ only in their input, so a checkpoint that does not record
    which arm, which encoder and which flags produced it is unattributable the
    moment there are two of them on disk.
    """
    torch.save({
        "arm": args.arm, "epoch": epoch, "metrics": metrics,
        "use_relpos": not args.no_relpos, "symmetric": args.symmetric,
        "encoder_ckpt": args.encoder_ckpt, "seed": args.seed,
        "unfrozen": args.unfreeze,
        "crop": args.crop, "pair_budget": args.pair_budget,
        "split": train._split_fingerprint(),
        "model": model.state_dict(),
        # Skipped ONLY for a frozen pretrained encoder, which is unchanged and
        # already on disk at encoder_ckpt -- storing 18.9M identical tensors in
        # every checkpoint would be duplicating it.
        #
        # Under --unfreeze the encoder is NOT that file any more, so it must be
        # saved or the fine-tuned weights are lost when the run ends.
        "provider": (None if args.arm == "pretrained" and not args.unfreeze
                     else provider.state_dict()),
        # Present only in *_last.pt. Without these a --resume would restart the
        # optimizer moments and the LR schedule, which is a warm start wearing a
        # resume's name -- and an LR discontinuity mid-cosine is not recoverable.
        "opt": opt.state_dict() if opt is not None else None,
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best,
    }, path)


def maybe_resume(path, args, model, provider, opt, sched, scaler):
    """Restore a run from *_last.pt. Returns (start_epoch, best_so_far).

    Refuses to resume across an ARM change: the two arms have different providers
    and different parameter groups, so the optimizer state would be meaningless
    even where the shapes happen to line up.
    """
    if not path:
        return 1, -1.0
    if not os.path.exists(path):
        raise SystemExit(f"--resume: no such file {path!r}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["arm"] != args.arm:
        raise SystemExit(f"--resume: checkpoint is arm {ck['arm']!r}, this run is "
                         f"{args.arm!r}. Resuming across arms is not meaningful.")
    if ck.get("opt") is None:
        raise SystemExit(f"--resume: {path!r} has no optimizer state. It is probably "
                         f"a *_best.pt; resume from *_last.pt instead.")
    model.load_state_dict(ck["model"])
    if ck["provider"] is not None:
        provider.load_state_dict(ck["provider"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    print(f"  resumed from {path} at epoch {ck['epoch']}, best P@L/5 {ck['best']:.3f}")
    return ck["epoch"] + 1, ck["best"]


def estimate_weights(loader, device, n_batches=WEIGHT_EST_BATCHES):
    """pos_weight and long_weight over a sample of real batches.

    Estimated ONCE and then held fixed, rather than recomputed per batch: a
    per-batch weight makes the loss a slightly different quantity every step, so
    the curve stops being comparable across epochs and early stopping reads noise.
    """
    pos = neg = long_ = 0.0
    for b, batch in enumerate(loader):
        if b >= n_batches:
            break
        _, mask, maps = train.to_device(batch)
        pm_all, pm_long = pair_mask_for(mask, 0), pair_mask_for(mask, MIN_SEP)
        p = (maps * pm_all).sum().item()
        pos += p
        neg += pm_all.sum().item() - p
        long_ += (maps * pm_long).sum().item()
    short = pos - long_
    return (torch.tensor(max(neg / max(pos, 1.0), 1.0), device=device),
            max(short / max(long_, 1.0), 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Contact predictor training")
    ap.add_argument("--arm", required=True,
                    choices=["pretrained", "scratch", "random"],
                    help="pretrained = frozen z_seq from --encoder-ckpt; scratch = "
                         "a 21-row embedding lookup; random = the SAME encoder "
                         "architecture at random init, frozen. No arm answers "
                         "anything alone: pretrained-vs-random isolates pretraining "
                         "(identical depth and capacity, only the weights differ), "
                         "pretrained-vs-scratch asks whether it beats the trivial "
                         "input, and random-vs-scratch says how much was just depth")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="Barlow Twins checkpoint for --arm pretrained")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--pair-budget", type=int, default=PAIR_BUDGET,
                    help="max padded pairs (B * L_max^2) per batch. THE memory "
                         "knob -- halve it on OOM before touching --crop")
    ap.add_argument("--crop", type=int, default=CROP,
                    help="0 (default) trains on whole proteins. A positive value "
                         "is an OOM fallback and caps what can be learned: pairs "
                         "further apart than this are never a training target")
    ap.add_argument("--symmetric-pair", action="store_true", dest="symmetric",
                    help="drop the column-attention pass. Exact, not an "
                         "approximation, when the pair tensor is symmetric: ~30%% "
                         "less memory and half the attention work, at the cost of "
                         "never holding an asymmetric feature")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="store activations instead of recomputing them. ~30%% "
                         "faster and ~3x the memory; only for short chains")
    ap.add_argument("--no-relpos", action="store_true",
                    help="drop the |i-j| embedding. The control that says how much "
                         "of the score came from the offset table rather than the "
                         "sequence")
    ap.add_argument("--unfreeze", action="store_true",
                    help="--arm pretrained only: fine-tune the encoder instead of "
                         "freezing it. The PRODUCT setting -- usually better contact "
                         "maps, but it does not preserve the joint embedding (nothing "
                         "holds z_seq in z_map's space any more) and a win no longer "
                         "isolates the representation from 18.9M extra trainable "
                         "parameters. Run it alongside the frozen arm: the GAP "
                         "between them measures how much the representation was "
                         "already carrying")
    ap.add_argument("--encoder-lr", type=float, default=None,
                    help="LR for pretrained weights under --unfreeze (default: "
                         "--lr / 10). A single LR across both lets large early "
                         "updates wreck the pretrained weights while the predictor "
                         "is still random")
    ap.add_argument("--eval-proteins", type=int, default=EVAL_PROTEINS,
                    help="val proteins scored each epoch, drawn once and then "
                         "fixed. Eval is uncropped and one protein at a time, so "
                         "this is the main cost between epochs. Lowering it makes "
                         "P@L/5 noisier -- do not compare runs that used different "
                         "values")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop each epoch after N steps (0 = whole epoch). For "
                         "validating a config end to end -- including eval, "
                         "checkpointing and resume -- in a minute instead of "
                         "committing hours. NOT a training setting: the LR "
                         "schedule still spans the full steps/epoch, so a run "
                         "using this is not comparable to one that does not")
    ap.add_argument("--resume", default=None,
                    help="continue from a *_last.pt, restoring optimizer, LR "
                         "schedule and scaler as well as the weights. Pass the "
                         "SAME --epochs as the original run: the cosine schedule "
                         "anneals to zero at that number, so changing it puts a "
                         "discontinuity in the LR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./contact_ckpt")
    ap.add_argument("--max-eval-len", type=int, default=MAX_EVAL_LEN)
    ap.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                    help="one train step + a few eval proteins, then exit")
    train.add_override_args(ap)      # --data-dir/--device/--num-workers/--amp-dtype
    args = ap.parse_args()

    train.apply_cli_overrides(args)
    device = train.DEVICE
    train.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tr_loader, val_set, n_train, n_val = build_pair_loaders(args.seed,
                                                            args.pair_budget)
    g = torch.Generator().manual_seed(args.seed)
    eval_idx = torch.randperm(len(val_set),
                              generator=g)[:args.eval_proteins].tolist()

    provider = build_input(args.arm, args.encoder_ckpt, device, args.unfreeze,
                           args.seed)
    model = ContactPredictor(use_relpos=not args.no_relpos,
                             symmetric=args.symmetric,
                             grad_checkpoint=not args.no_checkpoint).to(device)

    trainable = [p for p in list(model.parameters()) + list(provider.parameters())
                 if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in provider.parameters() if not p.requires_grad)

    steps_per_epoch = len(tr_loader)
    total_steps = args.epochs * steps_per_epoch
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr / 10
    opt = torch.optim.AdamW(param_groups(model, provider, args.lr, encoder_lr),
                            weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = train.USE_AMP and device.type != "cpu"
    amp_dtype = config.amp_dtype(train.AMP_DTYPE) if use_amp else None
    scaler = amp.GradScaler(device=device.type,
                            enabled=use_amp and amp_dtype is torch.float16)

    pos_w, long_w = estimate_weights(tr_loader, device)
    sizes = [len(b) for b in tr_loader.batch_sampler.batches]

    print("=" * 68)
    print(f"Contact predictor - ARM: {args.arm.upper()}")
    print(f"  device            : {device}")
    print(f"  encoder ckpt      : {args.encoder_ckpt or '(none -- scratch arm)'}")
    print(f"  train / val prot. : {n_train} / {n_val}")
    print(f"  eval pool         : {len(eval_idx)} val proteins, len <= "
          f"{args.max_eval_len}, fixed across epochs")
    print(f"  batching          : <={args.pair_budget:,} pairs -> "
          f"{min(sizes)}-{max(sizes)} proteins (median {int(np.median(sizes))}), "
          f"{steps_per_epoch} steps/epoch, total {total_steps}")
    print(f"  crop              : {'OFF -- whole proteins' if args.crop <= 0 else args.crop}")
    print(f"  pair track        : {'SYMMETRIC (no column pass)' if args.symmetric else 'asymmetric'}"
          f", checkpointing {'OFF' if args.no_checkpoint else 'ON'}")
    print(f"  relpos |i-j|      : {'OFF (control run)' if args.no_relpos else 'ON'}")
    print(f"  params            : {n_trainable/1e6:.2f}M trainable"
          + (f" + {n_frozen/1e6:.1f}M frozen encoder" if n_frozen else ""))
    if args.arm == "pretrained":
        print(f"  encoder           : "
              + (f"FINE-TUNED at lr {encoder_lr:.1e} (joint embedding NOT preserved)"
                 if args.unfreeze else "frozen, pretrained"))
    elif args.arm == "random":
        print(f"  encoder           : frozen, RANDOM INIT (depth control -- "
              f"pretrained minus this is what pretraining bought)")
    print(f"  pos_weight        : {pos_w.item():.1f}  (non-contacts per contact)")
    print(f"  long_weight       : {long_w:.2f}  (short-range positives per long)")
    print(f"  scored on         : P@L/5 at |i-j| > {MIN_SEP} and >= {STRICT_SEP}")
    print("=" * 68)

    def step_on(batch):
        ints, mask, maps = train.to_device(batch)
        with amp.autocast(device_type=device.type, dtype=amp_dtype,
                          enabled=amp_dtype is not None):
            x = provider(ints, mask)
        # Crop the REPRESENTATION, never the sequence: the encoder was trained on
        # whole chains and a truncated one changes what z_seq means. With --crop 0
        # this is a no-op and the whole protein goes through.
        x, maps_c, mask_c = crop_pair(x.float(), maps, mask, args.crop)
        with amp.autocast(device_type=device.type, dtype=amp_dtype,
                          enabled=amp_dtype is not None):
            logits = model(x, mask_c)
        # Loss in fp32 whatever --amp-dtype says, for the reason barlow_twins does
        # the same: a weighted sum over ~L^2 terms is exactly what overflows fp16.
        return contact_loss(logits.float(), maps_c, mask_c, pos_w,
                            separation_weight(mask_c, MIN_SEP, long_w))

    if args.smoke_test:
        model.train()
        provider.train()
        loss = step_on(next(iter(tr_loader)))
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters()), "no gradient reached the predictor"
        if args.arm in ("pretrained", "random"):
            # The assertion FLIPS with --unfreeze, and both directions are worth
            # checking: a frozen encoder that quietly trains invalidates the whole
            # comparison, and an "unfrozen" encoder that gets no gradient is a
            # fine-tuning run that silently is not fine-tuning anything.
            #
            # The random arm must be frozen too, or it stops being a control on
            # pretraining and becomes a control on "a big trainable encoder".
            got = [n for n, p in provider.named_parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0]
            if args.unfreeze:
                assert got, "--unfreeze set but no gradient reached the encoder"
                print(f"  encoder IS being fine-tuned: {len(got)} tensors have "
                      f"gradients  OK")
            else:
                assert not got, f"frozen encoder received gradients: {got[:3]}"
                print("  frozen encoder received NO gradients  OK")
        m = evaluate(model, provider, val_set, eval_idx[:5], device, pos_w, long_w,
                     amp_dtype, args.max_eval_len)
        # Exercise the save path too. Everything above can pass and the run still
        # die 20 epochs in on a checkpoint field -- _split_fingerprint(), or the
        # provider state_dict that is None in one arm and a tensor in the other.
        probe_path = os.path.join(args.out_dir, f"smoke_{args.arm}.pt")
        save_checkpoint(probe_path, args, model, provider, 0, m)
        reloaded = torch.load(probe_path, map_location="cpu", weights_only=False)
        assert reloaded["arm"] == args.arm and "model" in reloaded
        os.remove(probe_path)
        print(f"  checkpoint save/reload  OK")
        print(f"smoke test passed: loss {loss.item():.4f}, eval on {m['n']} "
              f"proteins P@L/5 {m['p']:.3f} (distance-only {m['p_dist']:.3f})")
        return

    start_epoch, best = maybe_resume(args.resume, args, model, provider,
                                     opt, sched, scaler)
    reported_mem = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        provider.train()
        running, n_seen = 0.0, 0
        for step, batch in enumerate(tr_loader):
            loss = step_on(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            n_seen += 1
            if not reported_mem and device.type == "cuda":
                # MEASURED, unlike the estimate in contact_predictor's docstring,
                # which is derived from the code and known to be approximate.
                # Retune --pair-budget from this number.
                print(f"  peak GPU memory after one step: "
                      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                reported_mem = True
            if n_seen % LOG_EVERY == 0:
                print(f"  [epoch {epoch} step {step}] loss {running/n_seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            if args.max_steps and n_seen >= args.max_steps:
                break

        m = evaluate(model, provider, val_set, eval_idx, device, pos_w, long_w,
                     amp_dtype, args.max_eval_len)
        print(f"epoch {epoch:3d} | train {running/max(n_seen,1):.4f} "
              f"| val {m['loss']:.4f} "
              f"| P@L/5 >{MIN_SEP} {m['p']:.3f} (dist {m['p_dist']:.3f}) "
              f"| >={STRICT_SEP} {m['p_strict']:.3f} (dist {m['p_strict_dist']:.3f})")

        # Selected on the METRIC, not the loss. train.py selects on total val loss
        # and that picked the epoch with the WORST agreement on the 150k run
        # (FINDINGS). No reason to repeat it where the quantity actually cared
        # about is measured every epoch anyway.
        if m["p"] > best:
            best = m["p"]
            path = os.path.join(args.out_dir, f"{args.arm}_best.pt")
            save_checkpoint(path, args, model, provider, epoch, m)
            print(f"  new best P@L/5 {best:.3f} -> {path}")

        # Written EVERY epoch, best or not, and it carries optimizer/scheduler/
        # scaler state so --resume is an actual resume rather than a warm start.
        # best.pt cannot serve this: it is only written on an improvement, so a
        # crash after a long plateau would resume from far behind, with a fresh
        # optimizer and an LR schedule restarted at the wrong step.
        save_checkpoint(os.path.join(args.out_dir, f"{args.arm}_last.pt"),
                        args, model, provider, epoch, m,
                        opt=opt, sched=sched, scaler=scaler, best=best)

    print(f"\ndone. best val P@L/5 (|i-j| > {MIN_SEP}) = {best:.3f}")
    print("This number means nothing alone. Run the other arm with the same "
          "--seed and --epochs, and compare -- and check both against the "
          "distance-only column before comparing them to each other.")


if __name__ == "__main__":
    main()
