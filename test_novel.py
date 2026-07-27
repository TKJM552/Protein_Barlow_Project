"""Pull a protein the model has NEVER seen from RCSB, and test it end to end.

Everything in eval.py runs on the 4,966 structures the model was trained on --
even the held-out val split came from the same query and the same era of the PDB.
This fetches a structure that is not in the dataset at all, processes it through
the exact training pipeline, and reports how the two branches compare.

    python test_novel.py                  # one random novel protein
    python test_novel.py --n 5            # five of them
    python test_novel.py --pdb-id 8ABC    # a specific entry

Downloads are cached in --cache-dir so repeat runs are free. Needs the extra
dependencies in requirements-data.txt (biopython, requests, scipy).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import train
from seq_encoder import ProteinSequenceDataset
from compare_embeddings import embed, pooled, linear_cka

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE = "https://files.rcsb.org/download/{}.cif"


def known_ids(data_dir):
    """The 4-character IDs already in the processed dataset -- what to EXCLUDE."""
    if not os.path.isdir(data_dir):
        return set()
    return {os.path.splitext(f)[0].lower()
            for f in os.listdir(data_dir) if f.endswith(".npz")}


def candidate_ids(n_wanted, exclude, seed=0):
    """Ask RCSB for protein entries, then drop anything already in the dataset.

    Sorted by release date descending, so we pull the NEWEST structures -- these
    are the least likely to overlap the training set in sequence as well as in ID.
    """
    import requests
    payload = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match", "value": "Protein"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "less_or_equal", "value": 1000}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 2000},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                      "direction": "desc"}],
        },
    }
    r = requests.post(RCSB_SEARCH, json=payload, timeout=60)
    r.raise_for_status()
    ids = [h["identifier"].lower() for h in r.json().get("result_set", [])]
    fresh = [i for i in ids if i not in exclude]
    if not fresh:
        raise SystemExit("RCSB returned nothing outside your dataset")
    rng = np.random.default_rng(seed)
    return list(rng.choice(fresh, min(n_wanted, len(fresh)), replace=False))


def fetch_cif(pdb_id, cache_dir):
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.cif")
    if os.path.exists(path):
        return path
    r = requests.get(RCSB_FILE.format(pdb_id.upper()), timeout=120)
    if r.status_code != 200:
        return None
    with open(path, "w") as f:
        f.write(r.text)
    return path


def analyse(modules, random_modules, seq_ints, cmap, control_targets, device):
    """One protein -> the three numbers that matter."""
    pred, target = embed(modules, seq_ints, cmap, device)
    r_pred, r_target = embed(random_modules, seq_ints, cmap, device)

    cka = linear_cka(pred.float(), target.float())
    cka_rand = linear_cka(r_pred.float(), r_target.float())
    per_res = float(F.cosine_similarity(pred, target, dim=-1).mean())

    p_vec = pooled(pred)
    matched = float(p_vec @ pooled(target))
    sims = (control_targets @ p_vec).numpy()
    pct = 100.0 * float((matched > sims).mean())
    return cka, cka_rand, per_res, pct


def main():
    ap = argparse.ArgumentParser(description="Test the model on unseen RCSB structures")
    ap.add_argument("--n", type=int, default=1, help="how many novel proteins to test")
    ap.add_argument("--pdb-id", default=None, help="test one specific PDB entry")
    ap.add_argument("--ckpt", default=None, help="default: CKPT_DIR/best.pt")
    ap.add_argument("--cache-dir", default="./novel_cache")
    ap.add_argument("--n-controls", type=int, default=100,
                    help="dataset proteins to use as mismatched controls")
    ap.add_argument("--seed", type=int, default=0)
    train.add_override_args(ap)
    args = ap.parse_args()
    train.apply_cli_overrides(args)
    device = train.DEVICE
    ckpt = args.ckpt or os.path.join(train.CKPT_DIR, "best.pt")

    try:
        import requests  # noqa: F401
        from get_inputs_outputs import process_pdb_file
    except ImportError:
        sys.exit("needs biopython/requests/scipy:  pip install -r requirements-data.txt")

    # --- models: the trained one, plus an untrained control ---------------
    modules = train.build_modules()
    epoch, val_loss = train.load_checkpoint(ckpt, modules)
    train.set_seed(args.seed + 999)
    random_modules = train.build_modules()
    for M in (modules, random_modules):
        for m in M.values():
            m.eval()
    print(f"checkpoint : {ckpt}  (epoch {epoch}, val_loss {val_loss:.3f})")
    print(f"device     : {device}")

    # --- controls: pooled targets of proteins from the training dataset ---
    ds = ProteinSequenceDataset(train.DATA_DIR)
    rng = np.random.default_rng(args.seed)
    ctrl_idx = rng.choice(len(ds), min(args.n_controls, len(ds)), replace=False)
    control_targets = torch.stack([
        pooled(embed(modules, *ds[int(i)], device)[1]) for i in ctrl_idx
    ])
    print(f"controls   : {len(control_targets)} proteins from the dataset\n")

    # --- pick the novel structures ----------------------------------------
    seen = known_ids(train.DATA_DIR)
    if args.pdb_id:
        ids = [args.pdb_id.lower()]
        if ids[0] in seen:
            print(f"WARNING: {ids[0]} IS in your dataset -- not a novel test.\n")
    else:
        print(f"querying RCSB for structures not among your {len(seen)} ...")
        ids = candidate_ids(args.n * 3, seen, seed=args.seed)

    hdr = f"{'PDB':<7}{'L':>5}{'CKA':>8}{'CKA_rnd':>9}{'cos/res':>9}{'pctile':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))

    rows, tried = [], 0
    for pid in ids:
        if len(rows) >= args.n:
            break
        tried += 1
        path = fetch_cif(pid, args.cache_dir)
        if path is None:
            continue
        try:
            result = process_pdb_file(path)
        except Exception:
            continue
        if result is None:
            continue
        _, seq_ints, contact_map = result
        L = len(seq_ints)
        if L < 40 or L > 1000:          # same bounds the training data respects
            continue
        s = torch.as_tensor(seq_ints, dtype=torch.long)
        c = torch.as_tensor(contact_map, dtype=torch.float32)
        cka, cka_rand, per_res, pct = analyse(modules, random_modules, s, c,
                                              control_targets, device)
        rows.append((pid, L, cka, cka_rand, per_res, pct))
        print(f"{pid:<7}{L:>5}{cka:>8.3f}{cka_rand:>9.3f}{per_res:>9.3f}{pct:>7.1f}%")

    if not rows:
        sys.exit("could not process any structure -- try again or pass --pdb-id")

    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    print("-" * len(hdr))
    print(f"{'mean':<12}{a[:,0].mean():>8.3f}{a[:,1].mean():>9.3f}"
          f"{a[:,2].mean():>9.3f}{a[:,3].mean():>7.1f}%")

    print("\nHOW TO READ THIS")
    print("  CKA      : shared geometry between the sequence and structure branches.")
    print("             Compare against CKA_rnd -- that is the same measurement on an")
    print("             UNTRAINED model, i.e. what the architecture gives for free.")
    print("  cos/res  : raw per-residue cosine. Expected ~0 even when training worked;")
    print("             Barlow Twins never forces the two spaces into a shared basis.")
    print("  pctile   : how the protein's own structure ranks against mismatched ones.")
    print("             50% = chance = no protein-specific signature.")
    print()
    if a[:, 0].mean() > a[:, 1].mean() + 0.2:
        print(f"  -> Geometry transfers to UNSEEN proteins "
              f"({a[:,0].mean():.2f} vs {a[:,1].mean():.2f} untrained).")
    else:
        print("  -> No better than an untrained model on unseen proteins.")
    if a[:, 3].mean() < 65:
        print("  -> No protein-level identity, as expected from TEST 4.")


if __name__ == "__main__":
    main()
