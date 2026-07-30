"""Pull proteins the model has NEVER seen from RCSB, and test them end to end.

Everything in eval.py runs on the structures the model was trained on -- even the
held-out val split came from the same query and the same era of the PDB.
This fetches structures that are not in the dataset at all, processes them
through the exact training pipeline, and reports how the two branches compare.

    python test_novel.py                  # a FRESH random sample every run
    python test_novel.py --n 5            # five of them
    python test_novel.py --pdb-id 9p5z    # a specific entry
    python test_novel.py --seed 0         # reproducible sample

By default each run draws from a different random window of recent RCSB
releases, and deletes the structures it downloaded when it finishes -- so you
get new proteins every time and nothing accumulates on disk. Pass --keep to
retain the .cif files.

Needs the extra dependencies in requirements-data.txt (biopython, requests, scipy).
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

import train
from map_encoder import MAX_LEN
from seq_encoder import ProteinSequenceDataset, MIN_RESIDUES
from compare_embeddings import embed, embed_map
from eval import linear_cka, CMP_LEN

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE = "https://files.rcsb.org/download/{}.cif"

# Entries are drawn from the newest RELEASE_WINDOW structures. The training set
# came from an unsorted query dominated by 1990s-2000s entries (101m, 1a42, ...),
# so anything in this window postdates it by decades -- novelty by construction,
# on top of the explicit ID exclusion below.
RELEASE_WINDOW = 15000


def known_ids(data_dir):
    """The 4-character IDs already in the processed dataset -- what to EXCLUDE."""
    if not os.path.isdir(data_dir):
        return set()
    return {os.path.splitext(f)[0].lower()
            for f in os.listdir(data_dir) if f.endswith(".npz")}


def candidate_ids(n_wanted, exclude, rng):
    """Ask RCSB for recent protein entries, then drop anything already seen.

    The pagination offset is randomised per run, so two runs pull from different
    parts of the release history rather than re-testing the same proteins.
    """
    import requests
    rows = max(60, n_wanted * 10)
    start = int(rng.integers(0, max(1, RELEASE_WINDOW - rows)))
    payload = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "operator": "exact_match", "value": "Protein"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "entity_poly.rcsb_sample_sequence_length",
                    "operator": "less_or_equal", "value": MAX_LEN}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                      "direction": "desc"}],
        },
    }
    r = requests.post(RCSB_SEARCH, json=payload, timeout=60)
    r.raise_for_status()
    ids = [h["identifier"].lower() for h in r.json().get("result_set", [])]
    fresh = [i for i in ids if i not in exclude]
    if not fresh:
        raise SystemExit("RCSB returned nothing outside your dataset -- try again")
    print(f"  drew {len(fresh)} candidates from release offset {start}")
    rng.shuffle(fresh)
    return fresh


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


def analyse(modules, random_modules, seq_ints, cmap, control_maps, device,
            cmp_len=CMP_LEN):
    """One novel protein -> the numbers that matter.

    Ranking uses CKA, NOT pooled cosine. Mean-pooling collapses different
    proteins onto near-identical vectors and destroys exactly the residue
    arrangement that identifies a protein -- see eval.py TEST 4.

    CRITICAL: every comparison in a ranking must use the SAME number of
    residues. CKA computed over 40 residues is not comparable to CKA over 260 --
    fewer samples inflates it -- so ranking mixed lengths lets short decoys beat
    the true structure. Everything here is truncated to exactly `n` residues,
    and controls shorter than that are skipped.
    """
    z_seq, z_map = embed(modules, seq_ints, cmap, device)
    r_seq, r_map = embed(random_modules, seq_ints, cmap, device)
    z_seq = z_seq.float(); z_map = z_map.float()

    # Reported CKA uses the whole protein (no ranking involved, so no truncation).
    cka_full = linear_cka(z_seq, z_map)
    cka_rand = linear_cka(r_seq.float(), r_map.float())
    per_res = float(F.cosine_similarity(z_seq, z_map, dim=-1).mean())

    # Ranking: fixed length, like for like.
    n = min(z_seq.shape[0], cmp_len)
    usable = [t for t in control_maps if t.shape[0] >= n]
    matched = linear_cka(z_seq[:n], z_map[:n])
    ctrl = np.array([linear_cka(z_seq[:n], t[:n]) for t in usable])
    rank = int((ctrl >= matched).sum()) + 1       # 1 = its own structure wins
    pct = 100.0 * float((matched > ctrl).mean())
    return cka_full, cka_rand, per_res, rank, pct, len(usable)


def main():
    ap = argparse.ArgumentParser(description="Test the model on unseen RCSB structures")
    ap.add_argument("--n", type=int, default=1, help="how many novel proteins to test")
    ap.add_argument("--pdb-id", default=None, help="test one specific PDB entry")
    ap.add_argument("--ckpt", default=None, help="default: CKPT_DIR/best.pt")
    ap.add_argument("--cache-dir", default="./novel_cache")
    ap.add_argument("--n-controls", type=int, default=40,
                    help="dataset proteins to rank against")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the sample for reproducibility (default: new each run)")
    ap.add_argument("--keep", action="store_true",
                    help="keep downloaded .cif files instead of deleting them")
    train.add_override_args(ap)
    args = ap.parse_args()
    train.apply_cli_overrides(args)
    device = train.DEVICE
    ckpt = args.ckpt or os.path.join(train.CKPT_DIR, "best.pt")

    # Only clean up a cache directory this run created. Without this, pointing
    # --cache-dir at an existing directory would delete it and everything in it.
    cache_is_ours = not os.path.isdir(args.cache_dir)

    # No --seed -> a genuinely different draw every run.
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    rng = np.random.default_rng(seed)
    print(f"sample seed: {seed}" + ("  (fixed)" if args.seed is not None else "  (random)"))

    try:
        import requests  # noqa: F401
        from get_inputs_outputs import process_pdb_file
    except ImportError:
        sys.exit("needs biopython/requests/scipy:  pip install -r requirements-data.txt")

    # --- models: the trained one, plus an untrained control ---------------
    modules = train.build_modules()
    epoch, val_loss, skipped = train.load_checkpoint(ckpt, modules)
    if skipped:
        # The whole point of this script is CKA between the sequence branch and the
        # MAP branch on unseen structures. A reinitialised tensor in either branch
        # turns that into a measurement of random weights.
        sys.exit(f"refusing to run: {len(skipped)} tensor(s) in '{ckpt}' do not match "
                 f"the current architecture (listed above). Retrain, or use a "
                 f"checkpoint from this architecture.")
    train.set_seed(999)
    random_modules = train.build_modules()
    for M in (modules, random_modules):
        for m in M.values():
            m.eval()
    print(f"checkpoint : {ckpt}  (epoch {epoch}, val_loss {val_loss:.3f})")
    print(f"device     : {device}")

    # --- controls: residue-level z_map from the training dataset -----------
    # Structure branch only: these are decoys ranked against a novel protein's
    # z_seq, so their own z_seq was computed and discarded.
    ds = ProteinSequenceDataset(train.DATA_DIR)
    ctrl_idx = rng.choice(len(ds), min(args.n_controls, len(ds)), replace=False)
    control_maps = [embed_map(modules, ds[int(i)][1], device).float() for i in ctrl_idx]
    print(f"controls   : {len(control_maps)} dataset proteins to rank against\n")

    # --- pick the novel structures ----------------------------------------
    seen = known_ids(train.DATA_DIR)
    if args.pdb_id:
        ids = [args.pdb_id.lower()]
        if ids[0] in seen:
            print(f"WARNING: {ids[0]} IS in your dataset -- not a novel test.\n")
    else:
        print(f"querying RCSB for structures not among your {len(seen)} ...")
        ids = candidate_ids(args.n, seen, rng)

    hdr = (f"{'PDB':<7}{'L':>5}{'CKA':>8}{'CKA_rnd':>9}{'cos/res':>9}"
           f"{'rank':>8}{'pctile':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = []
    for pid in ids:
        if len(rows) >= args.n:
            break
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
        # Exactly the bounds the training data respects, both imported rather than
        # written out: MIN_RESIDUES is the dataset's own floor, and MAX_LEN is the
        # number of columns the map encoder's seed projection has -- a longer chain
        # would trip its assert, not merely be atypical.
        if L < MIN_RESIDUES or L > MAX_LEN:
            continue
        s = torch.as_tensor(seq_ints, dtype=torch.long)
        c = torch.as_tensor(contact_map, dtype=torch.float32)
        cka, cka_rand, per_res, rank, pct, n_used = analyse(
            modules, random_modules, s, c, control_maps, device)
        rows.append((pid, L, cka, cka_rand, per_res, rank, pct, n_used))
        print(f"{pid:<7}{L:>5}{cka:>8.3f}{cka_rand:>9.3f}{per_res:>9.3f}"
              f"{rank:>5}/{n_used + 1}{pct:>7.1f}%")

    cleaned = cache_is_ours and not args.keep and not args.pdb_id
    if cleaned:
        shutil.rmtree(args.cache_dir, ignore_errors=True)

    if not rows:
        sys.exit("could not process any structure -- try again or pass --pdb-id")

    a = np.array([[r[2], r[3], r[4], r[6]] for r in rows])
    n_first = sum(1 for r in rows if r[5] == 1)
    print("-" * len(hdr))
    print(f"{'mean':<12}{a[:,0].mean():>8.3f}{a[:,1].mean():>9.3f}"
          f"{a[:,2].mean():>9.3f}{'':>8}{a[:,3].mean():>7.1f}%")

    print("\nHOW TO READ THIS")
    print("  CKA      : shared geometry between the sequence and structure branches.")
    print("             Compare against CKA_rnd -- the same measurement on an")
    print("             UNTRAINED model, i.e. what the architecture gives for free.")
    print("  cos/res  : raw per-residue cosine. Expected ~0 even when training worked;")
    print("             Barlow Twins never forces the two spaces into a shared basis.")
    print(f"  rank     : where its OWN structure placed among the decoys, by CKA")
    print(f"             (all comparisons over the first {CMP_LEN} residues).")
    print("             1 = correctly identified. Chance would be ~half way down.")
    print()
    if a[:, 0].mean() > a[:, 1].mean() + 0.2:
        print(f"  -> Geometry transfers to UNSEEN proteins "
              f"({a[:,0].mean():.2f} vs {a[:,1].mean():.2f} untrained).")
    else:
        print("  -> No better than an untrained model on unseen proteins.")
    mean_ctrl = float(np.mean([r[7] for r in rows]))
    print(f"  -> Own structure ranked #1 for {n_first}/{len(rows)} proteins "
          f"(chance {100/(mean_ctrl+1):.1f}%).")
    if cleaned:
        print(f"\n  ({args.cache_dir} deleted; pass --keep to retain downloads)")
    elif not cache_is_ours:
        print(f"\n  ({args.cache_dir} already existed, so it was left alone)")


if __name__ == "__main__":
    main()
