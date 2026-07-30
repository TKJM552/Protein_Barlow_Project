"""Get training structures from RCSB -- 150,000 of them, without hoarding .cif.

Two stages, deliberately split so the expensive one never has to run on a laptop:

    python get_files.py --ids      # ask RCSB *which* structures. 0.8 MB, ~12 s.
    python get_files.py --build    # fetch + process them. 2-5 h, run this ON THE POD.

`--ids` writes the query result -- one 4-character PDB ID per line -- to
$ID_LIST (./pdb_ids.txt). That file is small enough to commit, so the pod builds
from exactly the same list this machine resolved, no re-querying and no drift
from RCSB growing underneath the run.

`--build` streams each structure straight from RCSB into a .npz of
(sequence, seq_ints, contact_map) and throws the mmCIF away. Nothing is written
to disk except the .npz. That is the whole point of it:

                       raw .cif kept     .npz produced
    --download-cif       ~118 GB            ~0.5 GB     (legacy, needs both)
    --build                   0             ~0.5 GB

At 150,000 structures the old download-everything-first path would need 118 GB of
free disk before processing could start. The streamed build needs ~0.5 GB, and is
resumable -- it skips any ID that already has a .npz, so an interrupted run
continues where it stopped.

Requires the extras in requirements-data.txt (biopython, requests, scipy).
"""

import argparse
import gzip
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import requests

from config import (
    DATA_DIR,
    FETCH_WORKERS,
    ID_LIST,
    MAX_SEQ_LENGTH,
    MIN_RESIDUES,
    PDB_DIR,
    TARGET_STRUCTURES,
)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# .cif.gz, not .cif: ~10x smaller over the wire (50 KB vs 500 KB for a typical
# entry) and Biopython parses the gunzipped stream just as happily. Over 150,000
# structures that is ~22 GB transferred instead of ~200 GB.
FILE_URL = "https://files.rcsb.org/download/{}.cif.gz"

# RCSB rejects paginate.rows above this with a 400, so a 150,000-ID list is 15
# requests rather than one. Verified: rows=10000 -> 200, rows=20000 -> 400.
PAGE_ROWS = 10_000

# Ask for a few percent more IDs than TARGET_STRUCTURES. A handful of entries
# fail to parse, and the query filters on the SEQRES length of the longest
# polymer chain while what gets written is the longest RESOLVED amino-acid chain
# -- so a small number land outside [MIN_RESIDUES, MAX_SEQ_LENGTH] and are
# dropped at build time. The margin means the build still reaches the target.
OVERFETCH = 1.03

# Retry budget per structure. RCSB occasionally 500s or drops a connection under
# 16 concurrent workers; those are transient and worth retrying, a 404 is not.
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

PROGRESS_EVERY = 500

# Where --build records the IDs it could not use, one per line with the reason.
# Not inside DATA_DIR: the loader globs *.npz so a stray .txt would be harmless,
# but keeping build bookkeeping out of the dataset directory is cleaner.
FAILURE_LOG = "./build_failures.txt"


# ---------------------------------------------------------------------------
# Stage 1 -- which structures?
# ---------------------------------------------------------------------------
def build_query(start, rows, max_length=MAX_SEQ_LENGTH, min_length=MIN_RESIDUES):
    """One page of the RCSB search request.

    The length filter is on `rcsb_entry_info.polymer_monomer_count_maximum` --
    the LONGEST polymer chain in the entry -- not on
    `entity_poly.rcsb_sample_sequence_length` as this query used to be. The
    difference matters because return_type is "entry" while the old attribute is
    per-entity: an entry with a 120-residue protein and a 3,000-residue partner
    chain satisfied "some entity is <= 1000" and came back as a hit, and then
    get_inputs_outputs picks the LONGEST chain and had to throw it away. Filtering
    on the maximum makes the query itself guarantee the cap (238,948 hits vs
    248,653), which is why the build now loses almost nothing to over-length
    chains instead of ~0.7%.

    Sorted by release date ascending, with entry_id as a tiebreaker, for two
    reasons:

      1. Deep pagination needs a TOTAL order. Thousands of entries share a
         release date, so date alone leaves ties that the backend may order
         differently between requests -- which shows up as duplicate and missing
         IDs across page boundaries at start=140000.
      2. It puts the 150,000 at the OLD end of the PDB, leaving the ~84,000
         newest entries untouched. test_novel.py draws its never-seen proteins
         from the newest 15,000 releases, so that test stays honest by
         construction rather than by an ID exclusion list that would have to
         grow to 150,000 entries.
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_monomer_count_maximum",
                        "operator": "less_or_equal",
                        "value": max_length,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_monomer_count_maximum",
                        "operator": "greater_or_equal",
                        "value": min_length,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [
                {"sort_by": "rcsb_accession_info.initial_release_date",
                 "direction": "asc"},
                {"sort_by": "rcsb_entry_container_identifiers.entry_id",
                 "direction": "asc"},
            ],
        },
    }


def _post_search(payload, session=None):
    post = (session or requests).post
    for attempt in range(MAX_RETRIES):
        try:
            response = post(SEARCH_URL, json=payload, timeout=300)
            if response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()
                return response.json()
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(RETRY_BACKOFF ** attempt)
    raise RuntimeError(f"RCSB search failed after {MAX_RETRIES} attempts")


def count_matches(max_length=MAX_SEQ_LENGTH, min_length=MIN_RESIDUES):
    """How many entries the query matches, without pulling any IDs (rows=0)."""
    payload = build_query(0, 0, max_length, min_length)
    return _post_search(payload)["total_count"]


def fetch_ids(target=TARGET_STRUCTURES, max_length=MAX_SEQ_LENGTH,
              min_length=MIN_RESIDUES, overfetch=OVERFETCH):
    """Page through the search API until `target * overfetch` IDs are collected.

    Returns lowercase IDs (matching the .npz naming already in DATA_DIR) in query
    order. Raises if RCSB matches fewer entries than asked for, rather than
    quietly building a smaller dataset than requested.
    """
    session = requests.Session()
    wanted = int(target * overfetch)

    total = count_matches(max_length, min_length)
    print(f"RCSB matches {total:,} entries with a longest chain in "
          f"[{min_length}, {max_length}] residues")
    if total < target:
        raise RuntimeError(
            f"asked for {target:,} structures but the query only matches "
            f"{total:,}. Lower --target, or widen MAX_SEQ_LENGTH/MIN_RESIDUES."
        )
    wanted = min(wanted, total)

    ids, seen = [], set()
    while len(ids) < wanted:
        start = len(ids)
        rows = min(PAGE_ROWS, wanted - start)
        data = _post_search(build_query(start, rows, max_length, min_length), session)
        page = [hit["identifier"].lower() for hit in data.get("result_set", [])]
        if not page:
            break  # ran off the end of the result set

        # The sort is a total order, so pages should not overlap -- but dedupe
        # anyway and report it, because a silent duplicate would show up much
        # later as a dataset that never reaches its target count.
        fresh = [i for i in page if i not in seen]
        if len(fresh) != len(page):
            print(f"  note: dropped {len(page) - len(fresh)} duplicate IDs at "
                  f"start={start}")
        seen.update(fresh)
        ids.extend(fresh)
        print(f"  {len(ids):,} / {wanted:,} IDs", flush=True)

    print(f"Collected {len(ids):,} PDB IDs "
          f"(target {target:,} + {overfetch - 1:.0%} margin for build losses)")
    return ids


def write_ids(ids, path=ID_LIST):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    size_mb = os.path.getsize(path) / 1e6
    print(f"Wrote {len(ids):,} IDs to '{path}' ({size_mb:.1f} MB)")


def read_ids(path=ID_LIST):
    if not os.path.exists(path):
        raise SystemExit(
            f"No ID list at '{path}'. Run `python get_files.py --ids` first "
            f"(or pass --ids-file)."
        )
    with open(path) as fh:
        return [line.strip().lower() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Stage 2 -- fetch + process, keeping nothing
# ---------------------------------------------------------------------------
# One requests.Session per worker PROCESS, created in the initializer: it keeps
# the TLS connection to files.rcsb.org alive across structures, which is worth
# more than it sounds when the download is latency-bound (~255 ms/structure).
_SESSION = None


def _init_worker():
    global _SESSION
    _SESSION = requests.Session()


def _fetch_and_process(job):
    """Download one gzipped mmCIF, process it in memory, write the .npz.

    Runs in a worker process. Returns (pdb_id, category, detail): category is one
    of a FIXED set ("ok", "too short", "http error", "parse error", ...) so
    150,000 outcomes tally into a handful of lines, and detail carries the varying
    part. Never raises -- one bad entry out of 150,000 must not take the pool down.

    Processes rather than threads: MMCIFParser is pure Python, so parsing (~59 ms
    per structure, 2.5 CPU-hours over 150,000) is GIL-bound and would serialise
    across threads no matter how many were fetching.
    """
    pdb_id, out_dir, dist_threshold, max_length, min_length = job

    # Imported inside the worker so the parent process (and `--ids`, which needs
    # neither) does not pay for scipy/biopython.
    from get_inputs_outputs import check_length, process_cif, save_npz

    for attempt in range(MAX_RETRIES):
        try:
            response = _SESSION.get(FILE_URL.format(pdb_id.upper()), timeout=120)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                return pdb_id, "network error", type(exc).__name__
            time.sleep(RETRY_BACKOFF ** attempt)
            continue

        if response.status_code == 404:
            return pdb_id, "http error", "404 (obsolete or superseded entry)"
        if response.status_code != 200:
            if attempt == MAX_RETRIES - 1:
                return pdb_id, "http error", str(response.status_code)
            time.sleep(RETRY_BACKOFF ** attempt)
            continue
        break
    else:
        return pdb_id, "network error", f"no response in {MAX_RETRIES} attempts"

    try:
        # The .cif exists only as these bytes in RAM. TextIOWrapper because
        # MMCIFParser tokenises text, GzipFile because the wire format is gzip.
        handle = io.TextIOWrapper(
            gzip.GzipFile(fileobj=io.BytesIO(response.content)), encoding="utf-8"
        )
        result = process_cif(handle, pdb_id, dist_threshold=dist_threshold)
    except Exception as exc:
        return pdb_id, "parse error", f"{type(exc).__name__}: {exc}"

    if result is None:
        return pdb_id, "no amino acid chain", ""

    sequence, seq_ints, contact_map = result
    verdict = check_length(seq_ints, max_length=max_length, min_length=min_length)
    if verdict is not None:
        return pdb_id, verdict[0], verdict[1]

    save_npz(os.path.join(out_dir, f"{pdb_id}.npz"),
             sequence, seq_ints, contact_map)
    return pdb_id, "ok", ""


def stream_build(ids, out_dir=DATA_DIR, target=TARGET_STRUCTURES,
                 workers=FETCH_WORKERS, dist_threshold=8.0,
                 max_length=MAX_SEQ_LENGTH, min_length=MIN_RESIDUES,
                 failure_log=FAILURE_LOG):
    """Turn a list of PDB IDs into out_dir/*.npz, without keeping any mmCIF.

    Stops as soon as out_dir holds `target` structures, so the overfetch margin
    in the ID list costs nothing when the failure rate is low. Already-present
    .npz files count toward the target and are never re-downloaded, which is what
    makes an interrupted build resumable.
    """
    os.makedirs(out_dir, exist_ok=True)

    have = {os.path.splitext(f)[0] for f in os.listdir(out_dir)
            if f.endswith(".npz")}
    todo = [i for i in ids if i not in have]
    print(f"{len(have):,} structures already in '{out_dir}', "
          f"{len(todo):,} of {len(ids):,} IDs left to fetch, "
          f"target {target:,}")

    n_have = len(have)
    if n_have >= target:
        print("Target already met -- nothing to do.")
        return n_have, []
    if not todo:
        print(f"WARNING: ID list exhausted at {n_have:,} of {target:,}. "
              f"Re-run --ids with a larger --target.")
        return n_have, []

    jobs = [(i, out_dir, dist_threshold, max_length, min_length) for i in todo]
    failures = []
    n_ok = 0
    started = time.time()

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        results = pool.map(_fetch_and_process, jobs, chunksize=4)
        for n_done, (pdb_id, category, detail) in enumerate(results, start=1):
            if category == "ok":
                n_ok += 1
            else:
                failures.append((pdb_id, category, detail))

            if n_done % PROGRESS_EVERY == 0 or n_have + n_ok >= target:
                elapsed = time.time() - started
                rate = n_done / elapsed
                remaining = max(0, target - (n_have + n_ok))
                eta = remaining / rate / 60 if rate else float("inf")
                print(f"  {n_have + n_ok:,}/{target:,} built "
                      f"({len(failures):,} skipped) | {rate:.1f} struct/s | "
                      f"ETA {eta:.0f} min", flush=True)

            if n_have + n_ok >= target:
                # Drop the queued jobs instead of draining another 100k of them.
                pool.shutdown(wait=False, cancel_futures=True)
                break

    # Count the directory rather than the results we observed. pool.map hands
    # results back IN ORDER and in chunks, so by the time the target'th "ok" is
    # yielded, workers have already written up to a few hundred more .npz. Those
    # are real, usable structures -- the target is a floor, not a quota -- and
    # reporting n_have + n_ok here would under-report the dataset by that overshoot.
    total = sum(1 for f in os.listdir(out_dir) if f.endswith(".npz"))
    mins = (time.time() - started) / 60
    print(f"\nBuilt {total - n_have:,} new structures in {mins:.1f} min; "
          f"'{out_dir}' now holds {total:,}.")
    if failures:
        print(f"{len(failures):,} IDs skipped. Reasons:")
        tally = {}
        for _, category, _detail in failures:
            tally[category] = tally.get(category, 0) + 1
        for category, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"    {count:6,}  {category}")
        if failure_log:
            with open(failure_log, "w") as fh:
                for pdb_id, category, detail in failures:
                    fh.write(f"{pdb_id}\t{category}\t{detail}\n")
            print(f"    -> per-ID detail in '{failure_log}'. Network errors are "
                  f"worth a second --build pass; the length ones are not.")
    if total < target:
        print(f"WARNING: {total:,} < target {target:,}. Re-run --ids with a "
              f"larger --target (or --overfetch) and --build again.")
    return total, failures


# ---------------------------------------------------------------------------
# Legacy -- keep every .cif on disk
# ---------------------------------------------------------------------------
def download_structures(ids, output_dir=PDB_DIR):
    """Download every hit as mmCIF and KEEP it (~800 KB each, uncompressed).

    Only for rebuilding .npz repeatedly from the same structures, e.g. while
    tuning dist_threshold. For a one-shot dataset build use stream_build: at
    150,000 structures this path wants ~118 GB of free disk and the streamed one
    wants none.
    """
    from Bio.PDB import PDBList

    gb = len(ids) * 800e3 / 1e9
    print(f"About to download {len(ids):,} raw .cif files to '{output_dir}' "
          f"(~{gb:.0f} GB).")
    if gb > 10:
        reply = input("That is a lot of disk. Type 'yes' to continue: ")
        if reply.strip().lower() != "yes":
            raise SystemExit("Aborted.")

    os.makedirs(output_dir, exist_ok=True)
    PDBList().download_pdb_files(ids, pdir=output_dir, file_format="mmCif")
    print(f"Download complete. Now run `python get_inputs_outputs.py` to build "
          f"the .npz dataset from '{output_dir}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Default action is --ids: it writes a ~1 MB list and downloads "
               "no structures, so running this on a laptop by accident is cheap.",
    )
    parser.add_argument("--ids", action="store_true",
                        help="query RCSB and write the PDB ID list (default)")
    parser.add_argument("--build", action="store_true",
                        help="stream + process the IDs into .npz, keeping no .cif")
    parser.add_argument("--download-cif", action="store_true",
                        help="LEGACY: download raw .cif and keep them (~118 GB "
                             "at 150,000)")
    parser.add_argument("--count", action="store_true",
                        help="print how many entries the query matches, then exit")
    parser.add_argument("--target", type=int, default=TARGET_STRUCTURES,
                        help=f"structures to build (env: TARGET_STRUCTURES, "
                             f"default {TARGET_STRUCTURES:,})")
    parser.add_argument("--ids-file", default=ID_LIST,
                        help=f"where the ID list lives (env: ID_LIST, "
                             f"default {ID_LIST})")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"output dir for .npz (env: DATA_DIR, "
                             f"default {DATA_DIR})")
    parser.add_argument("--workers", type=int, default=FETCH_WORKERS,
                        help=f"parallel fetch+parse processes (env: FETCH_WORKERS, "
                             f"default {FETCH_WORKERS})")
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH,
                        help=f"residue cap (env: MAX_SEQ_LENGTH, default "
                             f"{MAX_SEQ_LENGTH}). Must stay <= map_encoder.MAX_LEN.")
    parser.add_argument("--min-length", type=int, default=MIN_RESIDUES,
                        help=f"residue floor (env: MIN_RESIDUES, default "
                             f"{MIN_RESIDUES})")
    parser.add_argument("--overfetch", type=float, default=OVERFETCH,
                        help=f"ID-list margin over --target (default {OVERFETCH})")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.count:
        print(f"{count_matches(args.max_length, args.min_length):,} entries match")
        return

    # --build/--download-cif reuse a committed ID list if there is one, so the
    # pod and this machine work from the identical set of structures.
    need_ids = args.ids or not (args.build or args.download_cif)
    if not need_ids and not os.path.exists(args.ids_file):
        print(f"No '{args.ids_file}' -- querying RCSB for one first.")
        need_ids = True

    if need_ids:
        ids = fetch_ids(args.target, args.max_length, args.min_length,
                        args.overfetch)
        write_ids(ids, args.ids_file)
    else:
        ids = read_ids(args.ids_file)
        print(f"Read {len(ids):,} IDs from '{args.ids_file}'")

    if args.build:
        stream_build(ids, args.data_dir, args.target, args.workers,
                     max_length=args.max_length, min_length=args.min_length)
    elif args.download_cif:
        download_structures(ids, PDB_DIR)
    else:
        print(f"\nNo structures downloaded (this is the default). Next, on the "
              f"machine that will train:\n"
              f"    python get_files.py --build\n"
              f"which streams {args.target:,} structures into '{args.data_dir}' "
              f"and keeps no .cif.")


if __name__ == "__main__":
    sys.exit(main())
