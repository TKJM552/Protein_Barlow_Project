"""Do a protein's SEQUENCE and its STRUCTURE land in the same place?

This is the model's central claim, tested on one protein at a time:

    sequence -> SequenceEncoder -> Predictor  -> pred    (B, L, 512)
    contact map ------------------> MapEncoder -> target  (B, L, 512)

If training worked, pred should resemble target for the SAME protein and not
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import train
from seq_encoder import ProteinSequenceDataset
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
    return (torch.as_tensor(seq_ints, dtype=torch.long),
            torch.as_tensor(contact_map, dtype=torch.float32),
            sequence)


@torch.no_grad()
def embed(modules, seq_ints, cmap, device):
    """One protein -> (pred, target), both (L, 512).

    A single protein needs no padding, so the mask is all-True. These are the
    PRE-expander representations -- the ones intended for downstream use.
    """
    L = seq_ints.shape[0]
    ints = seq_ints.unsqueeze(0).to(device)
    maps = cmap.unsqueeze(0).to(device)
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    seq_repr, _ = modules["sequence_encoder"](ints, mask)
    pred, _ = modules["predictor"](seq_repr, mask)
    target, _ = modules["map_encoder"](maps, mask)
    return pred[0], target[0]


def pooled(x):
    """(L, D) -> unit-norm (D,) mean over residues -- one vector per protein."""
    v = x.mean(0)
    return v / v.norm().clamp_min(1e-8)


def train_split_indices():
    """Reproduce train.py's 90/10 split so we can say which side a protein fell on."""
    full = ProteinSequenceDataset(train.DATA_DIR)
    n_total = len(full)
    n_val = int(train.VAL_FRACTION * n_total)
    gen = torch.Generator().manual_seed(train.SEED)
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
    epoch, val_loss = train.load_checkpoint(args.ckpt, modules)
    for m in modules.values():
        m.eval()
    print(f"checkpoint {args.ckpt}  (epoch {epoch}, val_loss {val_loss:.3f})")
    print(f"device     {device}\n")

    tr_idx, va_idx, ds = train_split_indices()

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

    pred, target = embed(modules, seq_ints, cmap, device)

    # --- 1. per-residue agreement -----------------------------------------
    per_res = F.cosine_similarity(pred, target, dim=-1)
    print("PER-RESIDUE cosine(pred_i, target_i)")
    print(f"  mean {per_res.mean():.3f}   median {per_res.median():.3f}   "
          f"min {per_res.min():.3f}   max {per_res.max():.3f}")

    print("  NOTE: ~0 is EXPECTED and is not a failure. Barlow Twins never asks")
    print("  pred_i to point the same way as target_i -- it pushes each through a")
    print("  SEPARATE expander and correlates DIMENSIONS across a batch. The two")
    print("  512-d spaces are free to use completely different bases.")

    # --- 1b. basis-independent similarity ---------------------------------
    # An untrained model is the only way to know whether a high CKA means
    # anything -- without it you cannot tell learning from architecture.
    train.set_seed(train.SEED + 999)
    random_modules = train.build_modules()
    for m in random_modules.values():
        m.eval()
    r_pred, r_target = embed(random_modules, seq_ints, cmap, device)

    cka = linear_cka(pred.float(), target.float())
    cka_rand = linear_cka(r_pred.float(), r_target.float())
    print(f"\nLINEAR CKA(pred, target)  -- do the two spaces share a geometry?")
    print(f"  trained     : {cka:.3f}")
    print(f"  random init : {cka_rand:.3f}   <- baseline: what the architecture gives free")
    print("  0 = unrelated | 1 = identical up to rotation/scale")

    # --- 2. protein-level, against mismatched controls --------------------
    p_vec, t_vec = pooled(pred), pooled(target)
    matched = float(p_vec @ t_vec)

    rng = np.random.default_rng(0)
    others = [i for i in range(len(ds)) if args.cif or i != args.index]
    others = rng.choice(others, min(args.n_controls, len(others)), replace=False)

    controls = []
    for i in others:
        o_seq, o_map = ds[int(i)]
        _, o_target = embed(modules, o_seq, o_map, device)
        controls.append(float(p_vec @ pooled(o_target)))
    controls = np.array(controls)

    print("\nPROTEIN-LEVEL cosine(pooled pred, pooled target)")
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
