"""Do a protein's SEQUENCE and its STRUCTURE land in the same place?

This is the model's central claim, tested on one protein at a time:

    sequence    -> SequenceEncoder -> z_seq  (B, L, 512)
    contact map -> MapEncoder ------> z_map  (B, L, 512)

If training worked, z_seq should resemble z_map for the SAME protein and not
for other proteins. The second half is what makes the number meaningful -- a
high similarity is worthless if every protein scores just as high, so every
run here compares the matched pair against a distribution of mismatched ones.

Usage:
    python compare_embeddings.py --index 0            # a protein from the dataset
    python compare_embeddings.py --cif some.cif       # any structure file
    python compare_embeddings.py --index 0 --ckpt checkpoints/best.pt

Reports whether the protein was in the training split, since a train protein
scoring well proves much less than a held-out one.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import train
from map_encoder import MAX_LEN
from seq_encoder import ProteinSequenceDataset, MIN_RESIDUES
# Single definition, shared with eval.py's TEST 6 -- see its docstring for why
# CKA rather than cosine is the right question for two unaligned spaces.
from eval import linear_cka


def load_protein_from_cif(path):
    """Any .cif -> (seq_ints, contact_map) using the SAME pipeline as training."""
    from get_inputs_outputs import process_pdb_file
    result = process_pdb_file(path)
    if result is None:
        raise SystemExit(f"no usable amino-acid chain found in {path}")
    sequence, seq_ints, contact_map = result
    # process_pdb_file applies NO length cap -- that lives in build_dataset -- but
    # the map encoder's seed projection has a fixed number of columns, so an
    # over-length chain has nowhere to go. Say so here rather than in an assert.
    if len(seq_ints) > MAX_LEN:
        raise SystemExit(
            f"{os.path.basename(path)}: longest chain is {len(seq_ints)} residues, "
            f"over the map encoder's MAX_LEN={MAX_LEN}. The training set is capped "
            f"at the same length, so this structure is outside what the model was "
            f"built for. Use a shorter structure, or raise MAX_LEN and retrain."
        )
    return (torch.as_tensor(seq_ints, dtype=torch.long),
            torch.as_tensor(contact_map, dtype=torch.float32),
            sequence)


def _full_mask(length, device):
    """A single protein needs no padding, so its mask is all-True."""
    return torch.ones(1, length, dtype=torch.bool, device=device)


@torch.no_grad()
def embed_seq(modules, seq_ints, device):
    """One protein's SEQUENCE -> z_seq (L, 512), pre-expander."""
    z_seq, _ = modules["sequence_encoder"](seq_ints.unsqueeze(0).to(device),
                                           _full_mask(seq_ints.shape[0], device))
    return z_seq[0]


@torch.no_grad()
def embed_map(modules, cmap, device):
    """One protein's CONTACT MAP -> z_map (L, 512), pre-expander."""
    z_map, _ = modules["map_encoder"](cmap.unsqueeze(0).to(device),
                                      _full_mask(cmap.shape[0], device))
    return z_map[0]


def embed(modules, seq_ints, cmap, device):
    """Both branches -> (z_seq, z_map), both (L, 512).

    Call embed_seq/embed_map directly when you only need one side. The control
    loops here and in test_novel.py rank many proteins' z_map against a single
    fixed z_seq, and running the sequence branch for every control doubled their
    cost for a result that was thrown away.
    """
    return embed_seq(modules, seq_ints, device), embed_map(modules, cmap, device)


def pooled(x):
    """(L, D) -> unit-norm (D,) mean over residues -- one vector per protein."""
    v = x.mean(0)
    return v / v.norm().clamp_min(1e-8)


def train_split_indices(split):
    """Reproduce the training run's 90/10 split, to say which side a protein fell on.

    `split` is the {seed, val_fraction, min_residues} block save_checkpoint stamps
    into every checkpoint. Reading it back matters: the split is a function of
    those three values, so a run trained with a non-default --seed would otherwise
    be labelled TRAIN/VAL from this module's defaults and be silently wrong.
    Checkpoints written before the stamp existed fall back to today's defaults.
    """
    full = ProteinSequenceDataset(
        train.DATA_DIR, min_residues=split.get("min_residues", MIN_RESIDUES))
    n_total = len(full)
    n_val = int(split.get("val_fraction", train.VAL_FRACTION) * n_total)
    gen = torch.Generator().manual_seed(split.get("seed", train.SEED))
    tr, va = random_split(full, [n_total - n_val, n_val], generator=gen)
    return set(tr.indices), set(va.indices), full


def main():
    ap = argparse.ArgumentParser(description="Compare sequence vs structure embeddings")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", type=int, help="index into the processed dataset")
    src.add_argument("--cif", help="path to any .cif structure file")
    ap.add_argument("--ckpt", default=os.path.join(train.CKPT_DIR, "best.pt"))
    ap.add_argument("--n-controls", type=int, default=200,
                    help="how many OTHER proteins to compare against")
    train.add_override_args(ap)
    args = ap.parse_args()
    train.apply_cli_overrides(args)
    device = train.DEVICE

    # --- model ------------------------------------------------------------
    modules = train.build_modules()
    ckpt = train.read_checkpoint(args.ckpt)
    epoch, val_loss, skipped = train.load_checkpoint(ckpt, modules, label=args.ckpt)
    for m in modules.values():
        m.eval()
    print(f"checkpoint {args.ckpt}  (epoch {epoch}, val_loss {val_loss:.3f})")
    if skipped:
        # This script compares map-encoder outputs across proteins; a reinitialised
        # tensor anywhere in that branch makes every number below meaningless.
        sys.exit(f"refusing to run: {len(skipped)} tensor(s) in '{args.ckpt}' do not "
                 f"match the current architecture (listed above), so the embeddings "
                 f"this compares would be partly random. Retrain, or use a "
                 f"checkpoint from this architecture.")
    print(f"device     {device}\n")

    if "split" not in ckpt:
        print(f"NOTE: this checkpoint predates the split stamp, so TRAIN/VAL below "
              f"assumes train.py's current defaults (seed {train.SEED}, val_fraction "
              f"{train.VAL_FRACTION}, min_residues {MIN_RESIDUES}). If the run used "
              f"anything else, the label is wrong.\n")
    tr_idx, va_idx, ds = train_split_indices(ckpt.get("split", {}))

    # --- the protein under test -------------------------------------------
    if args.cif:
        seq_ints, cmap, sequence = load_protein_from_cif(args.cif)
        label = os.path.basename(args.cif)
        split = "NOT in the dataset (truly novel)"
    else:
        seq_ints, cmap = ds[args.index]
        label = os.path.basename(ds.paths[args.index])
        split = ("TRAIN (model saw this)" if args.index in tr_idx
                 else "VAL (held out)")

    L = seq_ints.shape[0]
    n_contacts = int((cmap > 0.5).sum() - L) // 2
    print(f"protein    {label}")
    print(f"length     {L} residues, {n_contacts} contacts")
    print(f"split      {split}\n")

    z_seq, z_map = embed(modules, seq_ints, cmap, device)

    # --- 1. per-residue agreement -----------------------------------------
    per_res = F.cosine_similarity(z_seq, z_map, dim=-1)
    print("PER-RESIDUE cosine(z_seq_i, z_map_i)")
    print(f"  mean {per_res.mean():.3f}   median {per_res.median():.3f}   "
          f"min {per_res.min():.3f}   max {per_res.max():.3f}")

    print("  NOTE: ~0 is EXPECTED and is not a failure. Barlow Twins never asks")
    print("  z_seq_i to point the same way as z_map_i -- it pushes each through a")
    print("  SEPARATE expander and correlates DIMENSIONS across a batch. The two")
    print("  512-d spaces are free to use completely different bases.")

    # --- 1b. basis-independent similarity ---------------------------------
    # An untrained model is the only way to know whether a high CKA means
    # anything -- without it you cannot tell learning from architecture.
    train.set_seed(train.SEED + 999)
    random_modules = train.build_modules()
    for m in random_modules.values():
        m.eval()
    r_seq, r_map = embed(random_modules, seq_ints, cmap, device)

    cka = linear_cka(z_seq.float(), z_map.float())
    cka_rand = linear_cka(r_seq.float(), r_map.float())
    print(f"\nLINEAR CKA(z_seq, z_map)  -- do the two spaces share a geometry?")
    print(f"  trained     : {cka:.3f}")
    print(f"  random init : {cka_rand:.3f}   <- baseline: what the architecture gives free")
    print("  0 = unrelated | 1 = identical up to rotation/scale")

    # --- 2. protein-level, against mismatched controls --------------------
    p_vec, t_vec = pooled(z_seq), pooled(z_map)
    matched = float(p_vec @ t_vec)

    rng = np.random.default_rng(0)
    others = [i for i in range(len(ds)) if args.cif or i != args.index]
    others = rng.choice(others, min(args.n_controls, len(others)), replace=False)

    # Only the STRUCTURE branch is needed for the controls -- they are ranked
    # against this protein's fixed z_seq.
    controls = np.array([
        float(p_vec @ pooled(embed_map(modules, ds[int(i)][1], device)))
        for i in others
    ])

    print("\nPROTEIN-LEVEL cosine(pooled z_seq, pooled z_map)")
    print(f"  matched  (its OWN structure)      : {matched:+.4f}")
    print(f"  mismatched ({len(controls)} others)  : "
          f"{controls.mean():+.4f} +/- {controls.std():.4f}")
    beat = int((matched > controls).sum())
    print(f"  matched beats {beat}/{len(controls)} controls "
          f"({100*beat/len(controls):.1f}th percentile)")
    z = (matched - controls.mean()) / max(controls.std(), 1e-8)
    print(f"  z-score vs controls               : {z:+.2f}")

    print("\nVERDICT")
    if beat / len(controls) > 0.95 and z > 2:
        print("  Sequence embedding is specifically close to ITS OWN structure.")
    elif beat / len(controls) > 0.75:
        print("  Weakly protein-specific -- above most controls, but not decisively.")
    else:
        print("  NOT protein-specific: its own structure is no closer than a random")
        print("  one. The absolute cosine above is meaningless without this check --")
        print("  it just means all proteins map to a similar region.")


if __name__ == "__main__":
    main()
