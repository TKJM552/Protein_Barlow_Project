import os
import warnings
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial.distance import cdist

from config import PDB_DIR, DATA_DIR

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

    Picks the longest chain made of standard amino acids with a resolved CA
    atom (this drops waters/ligands and any other hetero residues).

    Returns (sequence, seq_ints, contact_map) or None if no usable chain
    is found.
    """
    structure_id = os.path.splitext(os.path.basename(cif_path))[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = _parser.get_structure(structure_id, cif_path)

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


def build_dataset(pdb_dir=PDB_DIR, out_dir=DATA_DIR,
                  dist_threshold=8.0, max_length=1000):
    """Process every .cif file in pdb_dir and save (sequence, seq_ints,
    contact_map) triples as one .npz per structure in out_dir.

    Chains longer than max_length are skipped, so every processed chain has
    length <= max_length (the RCSB query filters on entity length, but the
    longest RESOLVED chain in a structure can differ, so we enforce the cap
    here rather than trusting the query alone).
    """
    os.makedirs(out_dir, exist_ok=True)

    cif_files = sorted(f for f in os.listdir(pdb_dir) if f.endswith(".cif"))
    n_ok, n_failed, n_toolong = 0, 0, 0

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
        if len(seq_ints) > max_length:
            print(f"[{structure_id}] chain length {len(seq_ints)} > max_length "
                  f"{max_length}; skipping")
            n_toolong += 1
            continue

        out_path = os.path.join(out_dir, f"{structure_id}.npz")
        np.savez_compressed(
            out_path,
            sequence=sequence,
            seq_ints=seq_ints,
            contact_map=contact_map,
        )
        n_ok += 1

    print(f"Processed {n_ok} chains successfully, {n_failed} failed, "
          f"{n_toolong} over-length (>{max_length}). Saved to '{out_dir}'")


if __name__ == "__main__":
    build_dataset()
