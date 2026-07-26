"""Runtime configuration -- everything that differs between a laptop and a GPU pod.

No other file in this codebase hardcodes a filesystem location or a device, so
moving to a pod is just:

    export DATA_DIR=/workspace/processed_dataset
    export CKPT_DIR=/workspace/checkpoints
    export NUM_WORKERS=8
    python train.py

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
# Paths -- the two halves of the data pipeline, plus checkpoint output
# ---------------------------------------------------------------------------
#   get_files.py           -> PDB_DIR   raw .cif from RCSB   (~3.9 GB, NOT in git)
#   get_inputs_outputs.py  -> DATA_DIR  .npz training inputs (~23 MB, IS in git)
#   train.py               -> CKPT_DIR  ~500 MB per checkpoint (NOT in git)
PDB_DIR = os.environ.get("PDB_DIR", "./pdb_dataset")
DATA_DIR = os.environ.get("DATA_DIR", "./processed_dataset")
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
BATCH_SIZE = _env_int("BATCH_SIZE", 16)

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
