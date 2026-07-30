import os
import warnings
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial.distance import cdist

from config import PDB_DIR, DATA_DIR, MAX_SEQ_LENGTH, MIN_RESIDUES

#map 3-letter amino acid codes to 1-letter codes
AA_Mapping = {
    'ALA':'A',
    'CYS':'C',
    'ASP':'D',
    'GLU':'E',
    'PHE':'F',
    'GLY':'G',
    'HIS':'H',
    'ILE':'I',
    'LYS':'K',
    'LEU':'L',
    'MET':'M',
    'ASN':'N',
    'PRO':'P',
    'GLN':'Q',
    'ARG':'R',
    'SER':'S',
    'THR':'T',
    'VAL':'V',
    'TRP':'W',
    'TYR':'Y'
}

#amino acid vocabulary for AA to integer mapping
AA_Vocab = "ACDEFGHIKLMNPQRSTVWY"

#map amino acids to integers
aa_to_int = {aa: i+1 for i, aa in enumerate(AA_Vocab)}

_parser = MMCIFParser(QUIET=True)

def process_pdb_file(cif_path, dist_threshold=8.0):
    """Extract the amino acid sequence and CA-CA contact map for a .cif file.

    Returns (sequence, seq_ints, contact_map) or None if no usable chain
    is found.
    """
    structure_id = os.path.splitext(os.path.basename(cif_path))[0]
    return process_cif(cif_path, structure_id, dist_threshold=dist_threshold)


def process_cif(source, structure_id, dist_threshold=8.0):
    """Same as process_pdb_file, but `source` may be a path OR an open handle.

    The handle form is what makes the streaming build possible: get_files.py
    hands this an in-memory text stream wrapping a gunzipped HTTP response, so a
    150,000-structure dataset can be built without ever writing a .cif to disk
    (~117 GB avoided). Biopython's MMCIF2Dict accepts either, via as_handle.

    Picks the longest chain made of standard amino acids with a resolved CA
    atom (this drops waters/ligands and any other hetero residues).

    Returns (sequence, seq_ints, contact_map) or None if no usable chain
    is found.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = _parser.get_structure(structure_id, source)

    model = next(structure.get_models())

    best_residues = []
    for chain in model:
        residues = [
            res for res in chain
            if res.get_resname() in AA_Mapping and 'CA' in res
        ]
        if len(residues) > len(best_residues):
            best_residues = residues

    if not best_residues:
        return None

    sequence = ''.join(AA_Mapping[res.get_resname()] for res in best_residues)
    seq_ints = np.array([aa_to_int[aa] for aa in sequence], dtype=np.int64)

    coords = np.array([res['CA'].get_coord() for res in best_residues])
    dist_matrix = cdist(coords, coords)
    contact_map = (dist_matrix <= dist_threshold).astype(np.int8)

    return sequence, seq_ints, contact_map


def save_npz(out_path, sequence, seq_ints, contact_map):
    """Write one structure's .npz ATOMICALLY (temp file + rename).

    A 150,000-structure build is interrupted sooner or later -- Ctrl+C, a dropped
    connection, a pod restart. Writing in place would leave a truncated .npz that
    ProteinSequenceDataset then dies on, and the file exists so a resumed build
    would skip it. Rename is atomic on POSIX, so every file in out_dir is either
    absent or complete, which is what makes `--build` safely resumable.
    """
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(
            fh,
            sequence=sequence,
            seq_ints=seq_ints,
            contact_map=contact_map,
        )
    os.replace(tmp_path, out_path)


def check_length(seq_ints, max_length=MAX_SEQ_LENGTH, min_length=MIN_RESIDUES):
    """Return None if the chain is keepable, else a (category, detail) pair.

    The RCSB query filters on the longest polymer chain's SEQRES length, but what
    lands in the .npz is the longest RESOLVED amino-acid chain, which can differ
    in both directions -- so the cap is re-checked here rather than trusted from
    the query. min_length exists so the build does not spend disk on fragments
    the loader would drop anyway (see config.MIN_RESIDUES).

    The category is a FIXED string and the varying part lives in detail, so
    get_files.stream_build can tally 150,000 outcomes into a handful of lines.
    Returning one interpolated message instead made every skip its own category.
    """
    n = len(seq_ints)
    if n > max_length:
        return "too long", f"{n} residues > max_length {max_length}"
    if n < min_length:
        return "too short", f"{n} residues < min_length {min_length}"
    return None


def build_dataset(pdb_dir=PDB_DIR, out_dir=DATA_DIR, dist_threshold=8.0,
                  max_length=MAX_SEQ_LENGTH, min_length=MIN_RESIDUES):
    """Process every .cif file already on disk in pdb_dir into out_dir/*.npz.

    This is the LEGACY half of the pipeline, and only useful together with
    `get_files.py --download-cif`. The current path is `get_files.py --build`,
    which streams each structure from RCSB straight into out_dir and never keeps
    the .cif -- at 150,000 structures that is the difference between ~0.5 GB and
    ~118 GB of disk. This function stays because it is the only way to rebuild
    .npz from .cif you already have, e.g. after changing dist_threshold.
    """
    os.makedirs(out_dir, exist_ok=True)

    cif_files = sorted(f for f in os.listdir(pdb_dir) if f.endswith(".cif"))
    n_ok, n_failed, n_skipped = 0, 0, 0

    for fname in cif_files:
        structure_id = os.path.splitext(fname)[0]
        cif_path = os.path.join(pdb_dir, fname)

        try:
            result = process_pdb_file(cif_path, dist_threshold=dist_threshold)
        except Exception as e:
            print(f"[{structure_id}] failed to parse: {e}")
            n_failed += 1
            continue

        if result is None:
            print(f"[{structure_id}] no usable amino acid chain found")
            n_failed += 1
            continue

        sequence, seq_ints, contact_map = result
        verdict = check_length(seq_ints, max_length=max_length,
                               min_length=min_length)
        if verdict is not None:
            print(f"[{structure_id}] {verdict[0]}: {verdict[1]}; skipping")
            n_skipped += 1
            continue

        save_npz(os.path.join(out_dir, f"{structure_id}.npz"),
                 sequence, seq_ints, contact_map)
        n_ok += 1

    print(f"Processed {n_ok} chains successfully, {n_failed} failed, "
          f"{n_skipped} outside [{min_length}, {max_length}] residues. "
          f"Saved to '{out_dir}'")


if __name__ == "__main__":
    build_dataset()
