"""Runtime configuration -- everything that differs between a laptop and a GPU pod.

No other file in this codebase hardcodes a filesystem location or a device, so
moving to a pod is just:

    export DATA_DIR=/root/processed_dataset      # local disk, NOT /workspace
    export CKPT_DIR=/workspace/checkpoints
    export NUM_WORKERS=8
    python get_files.py --build                  # 2-5 h, streams from RCSB
    python train.py

The dataset is too large to ship in git (TARGET_STRUCTURES below), so the build
step is now part of pod setup rather than something `git clone` covers. Only the
~1 MB list of PDB IDs it works from is committed.

Precedence for every value here: CLI flag (train.py / eval.py) > environment
variable > the default below. Defaults are the laptop values, so running with no
flags and no environment reproduces the original behaviour exactly.
"""

import os

import torch


# ---------------------------------------------------------------------------
# Env parsing helpers -- fail loudly on a typo instead of silently falling back
# ---------------------------------------------------------------------------
def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"environment variable {name}={raw!r} is not an integer")


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"environment variable {name}={raw!r} is not a number")


# ---------------------------------------------------------------------------
# Paths -- the data pipeline's outputs, plus checkpoint output
# ---------------------------------------------------------------------------
#   get_files.py --ids     -> ID_LIST   the RCSB query result (~1 MB, IS in git)
#   get_files.py --build   -> DATA_DIR  .npz training inputs (see TARGET_STRUCTURES)
#   train.py               -> CKPT_DIR  ~500 MB per checkpoint (NOT in git)
#
# PDB_DIR is only used by the LEGACY `get_files.py --download-cif` path, which
# keeps raw .cif on disk. At TARGET_STRUCTURES=150,000 that is ~117 GB, so the
# default build streams instead and never writes a .cif at all.
PDB_DIR = os.environ.get("PDB_DIR", "./pdb_dataset")
DATA_DIR = os.environ.get("DATA_DIR", "./processed_dataset")
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
ID_LIST = os.environ.get("ID_LIST", "./pdb_ids.txt")


# ---------------------------------------------------------------------------
# Dataset scale and shape
# ---------------------------------------------------------------------------
# How many usable structures to build. The RCSB query behind it currently matches
# 238,948 entries, so this can go higher; it is the count of .npz files that end
# up in DATA_DIR, not the number of IDs requested (get_files.py over-fetches a
# small margin to cover entries that fail to parse).
#
# Cost at 150,000, extrapolated from a measured random sample of 1,067 entries
# (mean chain 271 residues, mean .npz 3.0 KB) rather than guessed:
#   ~15 GB downloaded (gzipped mmCIF, streamed and discarded -- nothing kept)
#   ~0.5 GB of .npz on disk           (vs 23 MB at the old 5,000)
#   ~5 h to build at FETCH_WORKERS=16 on a home connection (7.7 structures/s,
#     network-bound -- only 220% CPU of 8 cores). Expect materially faster on a
#     pod, where parsing becomes the limit; measure the first 5,000 rather than
#     trusting this line.
#   ~50 s for ProteinSequenceDataset to scan it on LOCAL disk (see POD_SETUP.md --
#     on RunPod's network-backed /workspace this becomes ~2.5 h, which is the one
#     mistake that makes this dataset size unusable)
TARGET_STRUCTURES = _env_int("TARGET_STRUCTURES", 150_000)

# The length cap, and the single source of truth for it. Three places must agree
# and now all read this: the RCSB query (get_files.py) filters on the longest
# polymer chain, build time re-checks the RESOLVED chain it actually writes
# (get_inputs_outputs.py), and the contact-map encoder sizes its seed projection
# from it (map_encoder.MAX_LEN).
#
# WARNING: raising this changes the map encoder's ARCHITECTURE (the seed
# projection is 2*MAX_LEN-1 columns wide), so existing checkpoints will not load.
# Lowering it below the longest chain already in DATA_DIR makes train.py raise.
MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", 1000)

# Shortest chain worth training on -- see seq_encoder.MIN_RESIDUES for why 40.
# Enforced at BUILD time now as well as load time, so the 150,000 structures on
# disk are 150,000 *trainable* structures rather than 150,000 minus a 5% tail of
# 1-40 residue fragments the loader would silently drop.
MIN_RESIDUES = _env_int("MIN_RESIDUES", 40)

# Parallel fetch+parse processes for `get_files.py --build`. Processes, not
# threads: MMCIFParser is pure Python and GIL-bound (~59 ms/structure), while the
# download it waits on is latency-bound (~255 ms). 16 covers both on an 8-core
# box. Lower it if RCSB starts refusing connections.
FETCH_WORKERS = _env_int("FETCH_WORKERS", 16)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# TRAINING batches are formed by residue budget, not protein count -- see
# RESIDUES_PER_BATCH below. BATCH_SIZE is the fixed protein count used by
# everything else: eval.py's diagnostics and the per-module sanity checks.
BATCH_SIZE = _env_int("BATCH_SIZE", 16)

# The memory knob for training. train.py groups proteins of similar length and
# closes a batch when (proteins x longest chain in it) would exceed this, which
# is exactly the padded width collate_pad materialises -- so this bounds real
# worst-case memory rather than an average.
#
# 4096 is chosen to match what random batches of 16 already averaged on this
# dataset (median 4086 residues), so gradient statistics carry over from earlier
# runs while the padding waste (2.41x -> 1.00x) and peak attention memory (4.6x
# lower) do not. Halve it if you OOM; it is the first thing to lower.
#
# Do NOT set it below ~2048: Barlow Twins treats each residue as a sample and
# needs N > EXPANDER_DIM for the cross-correlation to be well conditioned
# (barlow_twins.py). It must also be >= the longest chain in the dataset.
RESIDUES_PER_BATCH = _env_int("RESIDUES_PER_BATCH", 4096)

# 0 = decode .npz files in the main process. That is fine on a laptop, but on a
# GPU it is usually the bottleneck -- the GPU sits idle while one CPU thread
# decompresses. 4-8 is a sensible starting point on a pod.
NUM_WORKERS = _env_int("NUM_WORKERS", 0)


# ---------------------------------------------------------------------------
# Device / mixed precision
# ---------------------------------------------------------------------------
# "fp16" needs a GradScaler (loss scaling) and is the default because it matches
# the original behaviour. "bf16" has fp32's exponent range, so it needs no scaler
# and cannot overflow the way fp16 can -- prefer it on Ampere or newer (A100,
# A10, L4, 30xx/40xx, H100).
AMP_DTYPE = os.environ.get("AMP_DTYPE", "fp16")

_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def amp_dtype(name=None):
    """Map an --amp-dtype / $AMP_DTYPE string to a torch dtype."""
    key = (name or AMP_DTYPE).lower()
    if key not in _AMP_DTYPES:
        raise ValueError(f"AMP_DTYPE must be one of {sorted(_AMP_DTYPES)}, got {key!r}")
    return _AMP_DTYPES[key]


def resolve_device(requested=None):
    """Return a torch.device: CLI flag wins, then $DEVICE, then cuda-if-available.

    Returns a torch.device (not the string "cuda") on purpose -- the training
    loop reads DEVICE.type for autocast and GradScaler.

    Apple Silicon (mps) is reachable via DEVICE=mps but is never auto-selected:
    it is slower than cpu for this model's masked attention and would silently
    change local debug runs.
    """
    name = requested or os.environ.get("DEVICE")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
