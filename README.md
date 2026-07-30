# Protein Barlow

A two-branch **Barlow Twins** model for proteins. A sequence branch reads amino
acids; a structure branch reads the CA-CA contact map. The objective makes the two
representations agree dimension-for-dimension, so the sequence encoder ends up
carrying structural information without ever seeing coordinates at inference.

```
DataLoader -> SequenceEncoder -> z_seq \
                                        Barlow Twins loss
              MapEncoder -----> z_map  /
```

The objective is **symmetric**: neither branch predicts the other and neither is a
"target", which is why the two are named `z_seq` and `z_map`. There is no
prediction head — the expanders inside the loss are already a per-residue MLP on
each branch, so a separate one added nothing they could not express.

This is **joint** training: one AdamW optimizer holds the sequence encoder, the
map encoder, and the Barlow Twins expanders. There is no frozen branch, no
stop-gradient, and no EMA — collapse is prevented purely by the loss's
off-diagonal redundancy term.

---

## Start here

- **[FINDINGS.md](FINDINGS.md)** — the live lab notebook. Currently states what
  has and has not been measured on the present architecture.
- **[FINDINGS_2026-07-26_jepa.md](FINDINGS_2026-07-26_jepa.md)** — the archived
  50-epoch run under the *previous* JEPA architecture: why it plateaued, the
  ranked next experiments, and the measurement traps. Its numbers do not describe
  the current model, but the traps section still applies to any run.
- **[POD_SETUP.md](POD_SETUP.md)** — the RunPod sequence that works, including
  the `/workspace` slowness that costs 5 minutes per invocation if ignored.

## Quickstart on a GPU pod

The dataset is **150,000 structures**, too large to ship in git, so building it is
the first step on the pod. What *is* committed is
[pdb_ids.txt](pdb_ids.txt) — the 154,500 PDB IDs the build works from — so the
build is reproducible and needs no second RCSB query.

```bash
git clone <your-repo-url>
cd Protein_Barlow_Project

# Most GPU images already ship a CUDA torch. If yours doesn't:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-data.txt      # needed to BUILD the dataset

# 1. Build the dataset. Streams each structure from RCSB, writes one .npz, and
#    discards the mmCIF -- ~0.5 GB lands on disk, not the ~118 GB of raw .cif.
#    Resumable: re-run it after an interruption and it picks up where it stopped.
python get_files.py --build --workers 16

# 2. Confirm the wiring on the GPU (a minute, no real training)
python train.py --smoke-test

# 3. Train
python train.py --num-workers 8 --amp-dtype bf16 --epochs 5
```

`--smoke-test` runs one train step and one val step, asserts the loss is finite,
and asserts gradients reach every module group. Run it first — it catches a
broken environment in seconds instead of at epoch 3.

**Budget for the new scale before starting a run.** Measured/extrapolated at
150,000 structures:

| | 5,000 structures (old) | 150,000 structures |
|---|---|---|
| build time | ~1 h, 3.9 GB of `.cif` kept | ~2–5 h, nothing kept |
| dataset on disk | 23 MB, 4,966 files | ~0.5 GB, 150,000 files |
| dataset scan at startup | ~1 s | ~40 s (local disk) |
| steps/epoch (4096 residues/batch) | 286 | ~9,640 |
| epoch time on a 4090 | ~1 min | ~35 min |

`EPOCHS = 50` was tuned for 286-step epochs; at 9,640 steps it is ~480,000 steps
and roughly a day of 4090 time. With 30× the data, start with `--epochs 5` and
read the val curve before committing to more — that is already 2.4× the gradient
steps the archived 50-epoch run took.

To train against data or checkpoints living outside the repo:

```bash
export DATA_DIR=/root/processed_dataset       # local disk -- see POD_SETUP.md
export CKPT_DIR=/workspace/checkpoints
python train.py
```

---

## Configuration

Every value that differs between a laptop and a pod lives in [config.py](config.py)
and is overridable. Precedence: **CLI flag > environment variable > default.**

| Env var | Flag | Default | Purpose |
|---|---|---|---|
| `DATA_DIR` | `--data-dir` | `./processed_dataset` | Processed `.npz` training inputs |
| `CKPT_DIR` | `--ckpt-dir` | `./checkpoints` | Checkpoint output |
| `PDB_DIR` | — | `./pdb_dataset` | Raw `.cif` downloads (dataset rebuild only) |
| `DEVICE` | `--device` | `cuda` if available, else `cpu` | `cuda`, `cuda:0`, `cpu`, `mps` |
| `NUM_WORKERS` | `--num-workers` | `0` | DataLoader workers — **set 4–8 on a GPU** |
| `RESIDUES_PER_BATCH` | `--residues-per-batch` | `4096` | **Training batch budget** — the memory knob |
| `BATCH_SIZE` | `--batch-size` | `16` | Proteins per batch for `eval.py`'s diagnostics only |
| `AMP_DTYPE` | `--amp-dtype` | `fp16` | `fp16` or `bf16` |

Training hyperparameters also have flags: `--epochs`, `--lr`, `--weight-decay`,
`--warmup-steps`, `--seed`, `--ckpt-every`, `--keep-epoch-ckpts`, `--resume`,
`--warm-start`, `--no-amp`. Run `python train.py --help` for the full list.

### Batching

Training batches are formed by **residue budget, not protein count.** Proteins are
grouped by length and a batch closes when `proteins × longest chain in it` would
exceed `--residues-per-batch`, which is exactly the padded width `collate_pad`
materializes.

This is worth knowing because it was ~50% of the training cost. Chains run 40–990
residues (median 218), so a random batch of 16 padded everything up to its longest
member and did **2.41× the useful work**. Bucketing brings that to 1.00× and cuts
worst-case attention memory 4.6×.

The budget is counted in residues rather than proteins because Barlow Twins treats
each *residue* as a sample and needs `N > EXPANDER_DIM = 2048` for the
cross-correlation to be well conditioned. Bucketing at a fixed 16 proteins/batch
would put 22% of batches below that bound; the residue budget holds `N` nearly
constant (median 3877), which is *steadier* than the random batching it replaced.
At the default it works out to 3–71 proteins per batch, median 12, 286 steps/epoch.

`--resume` refuses a checkpoint written by a different architecture: the saved
AdamW moments are shape-bound to the parameters that produced them. `--warm-start`
takes the weights that still fit, reinitialises the rest, and restarts the LR
schedule. Everything that loads a checkpoint reports exactly which tensors it had
to drop, and `eval.py` marks any run whose numbers come from partly-random weights.

### GPU notes

- **`--num-workers` matters.** The default of `0` decodes `.npz` files in the main
  process, which leaves the GPU idle between batches. Start at 8.
- **Prefer `--amp-dtype bf16` on Ampere or newer** (A100, A10, L4, H100, 30xx/40xx).
  bf16 has fp32's exponent range and needs no loss scaling. The `GradScaler` is
  enabled only for fp16. Note the loss itself is always computed in fp32 —
  `off_diag` routinely exceeds fp16's 65504 ceiling early in training, so
  `barlow_twins_core` disables autocast for that block. Without it, an fp16 run
  returns `inf`, every step gets skipped, and training silently does nothing.
- **If you hit OOM, lower `--residues-per-batch`** (not `--batch-size`, which no
  longer governs training). It bounds `proteins × longest chain` directly, so
  halving it halves peak activation memory. Keep it above ~2048 so each batch
  still yields more residue-samples than `EXPANDER_DIM`.
- **Checkpoints are ~500 MB each** (43M params × model + optimizer + scheduler
  state). `best.pt` plus a rolling `last.pt` is ~1 GB total; pass
  `--keep-epoch-ckpts` if you want every `--ckpt-every` snapshot kept instead
  (~5.5 GB over 50 epochs).

### Resuming

```bash
python train.py --resume $CKPT_DIR/last.pt
```

Restores module weights, optimizer, scheduler, and scaler state, re-forms the
batch schedule for the right epoch, then continues from the next epoch. `last.pt`
is written every `--ckpt-every` epochs and is the newest crash-resume point;
`best.pt` also resumes but rewinds to whenever validation last improved.

---

## Diagnostics

[eval.py](eval.py) answers "is it actually learning?" independently of training
loss. Each test prints the numbers you should expect if things are working *and*
if they are broken.

```bash
python eval.py --test all --ckpt $CKPT_DIR/best.pt
```

| Test | Question it answers |
|---|---|
| `overfit` | Can the model memorize 4 proteins? (Gradients flowing at all?) |
| `shuffled` | Does real sequence↔map pairing beat a deliberately mismatched one? |
| `collapse` | Are the representations degenerate (dead dims, low effective rank)? |
| `retrieval` | Is each protein's `z_seq` closest to *its own* `z_map`? |
| `probe` | Does a frozen encoder + linear probe beat random init and a distance-only baseline on long-range contacts? |
| `alignment` | Do the two branches share a geometry? (CKA vs an untrained model) |

Without `--ckpt` the model is randomly initialized, which is a meaningful run —
it should produce the "broken/near-chance" numbers and confirm the diagnostics are
calibrated. `probe` is the load-bearing one: it must beat **both** random init and
the distance-only baseline, or the encoder only learned "near in sequence = near in
space."

---

## Building the dataset

Two stages, split so the expensive one never has to run on a laptop:

```bash
pip install -r requirements-data.txt

python get_files.py --ids     # ask RCSB WHICH structures -> pdb_ids.txt (0.8 MB, ~12 s)
python get_files.py --build   # fetch + process them -> $DATA_DIR (~0.5 GB, hours)
```

`--ids` writes one PDB ID per line and downloads no structures, so it is safe
anywhere. `--build` streams each structure from RCSB as gzipped mmCIF, extracts
the chain in memory, writes the `.npz`, and throws the mmCIF away — **no `.cif`
ever touches disk.** At this scale that is the difference between 0.5 GB and
~118 GB, which is why the old download-everything-first path is now opt-in
(`--download-cif`, still there for re-deriving `.npz` at a different
`dist_threshold` from structures you already have).

`--build` is resumable: it skips any ID that already has a `.npz`, so an
interrupted run continues, and it stops as soon as `TARGET_STRUCTURES` is reached.

Each `.npz` holds `sequence` (string), `seq_ints` (`int64`, `1..20`, `0` = pad),
and `contact_map` (`int8`, `(L, L)`, symmetric, 1-diagonal, CA-CA ≤ 8 Å).

`MIN_RESIDUES = 40` is now applied at **build** time as well as load time. It is
the same floor `test_novel.py` applies to structures it pulls from RCSB, so the
model is trained and tested on one length distribution — and a fresh build spends
no disk on the sub-40-residue fragments the loader would silently drop (5% of the
old 5,000-structure dataset).

### Which 150,000, and why

The RCSB query matches 234,073 entries; the build takes the **oldest 150,000 by
release date**, ascending, with entry ID as a tiebreaker. Both parts are
deliberate:

- Deep pagination needs a **total** order. Thousands of entries share a release
  date, and ties can come back in a different order per request — which at
  `start=140000` shows up as duplicate and missing IDs across page boundaries.
- Taking the *old* end leaves the ~84,000 newest entries out of training, and
  that is the pool [test_novel.py](test_novel.py) draws its never-seen proteins
  from. Verified disjoint: of the 10,000 most recently released matching entries,
  **0** are in the training set.

Length limits are enforced twice. The query filters on
`rcsb_entry_info.polymer_monomer_count_maximum` — the **longest** polymer chain in
the entry — so every hit is within `[MIN_RESIDUES, MAX_SEQ_LENGTH]` = `[40, 1000]`.
The build then re-checks the chain it actually writes, because what lands in the
`.npz` is the longest *resolved* amino-acid chain and that can differ from SEQRES.
The old query filtered per-*entity* while returning *entries*, so an entry with a
120-residue protein next to a 3,000-residue partner chain counted as a hit and was
then discarded at build time; fixing that is why the ID list needs only a 3%
overfetch margin to land 150,000 usable structures.

Sizes, timings, and the length cap all live in [config.py](config.py)
(`TARGET_STRUCTURES`, `MAX_SEQ_LENGTH`, `MIN_RESIDUES`, `FETCH_WORKERS`), each
overridable by env var or CLI flag. `MAX_SEQ_LENGTH` is now the single source of
truth for the cap — `map_encoder.MAX_LEN` reads it — so raising it changes the map
encoder's weight shapes and invalidates existing checkpoints.

The query is not pinned to a snapshot, so re-running `--ids` months from now will
return a slightly different list as the PDB grows. `pdb_ids.txt` is committed so
that a rebuild reproduces *this* dataset exactly.

---

## Repository layout

| File | Role |
|---|---|
| [config.py](config.py) | Paths, device, workers, AMP — all env-overridable |
| [seq_encoder.py](seq_encoder.py) | Dataset, padding collate, token embedding, RoPE transformer encoder |
| [map_encoder.py](map_encoder.py) | Contact-map encoder; attention restricted to actual contacts |
| [barlow_twins.py](barlow_twins.py) | Expanders + the redundancy-reduction loss |
| [train.py](train.py) | Joint training loop, checkpointing, CLI |
| [eval.py](eval.py) | The six diagnostics above |
| [compare_embeddings.py](compare_embeddings.py) | One protein through both branches: CKA, per-residue cosine, ranking vs controls |
| [test_novel.py](test_novel.py) | Fetches unseen structures from RCSB and runs the same comparison |
| [get_files.py](get_files.py) | RCSB query (`--ids`) + streaming dataset build (`--build`) |
| [get_inputs_outputs.py](get_inputs_outputs.py) | `.cif` (path or stream) → sequence + contact map `.npz` |
| [pdb_ids.txt](pdb_ids.txt) | The 154,500 PDB IDs the build works from — committed, so builds reproduce |

Every module has a `__main__` sanity check; run any file directly to exercise it:

```bash
python seq_encoder.py    # shapes, symmetry, zero-padding, no padding leak
python map_encoder.py    # shapes, blindness, no padding leak, seed diversity,
                         # position leakage (both seed modes)
python barlow_twins.py   # grads on both sides, identity ≈ 0, collapse punished
```

### Not in git

`processed_dataset/` (~150,000 `.npz`, ~0.5 GB — rebuild it with
`python get_files.py --build`), `pdb_dataset/` (only written by the legacy
`--download-cif` path), `checkpoints/` (~500 MB per file, over GitHub's 100 MB
limit), and `.venv/`. See [.gitignore](.gitignore).

The 4,966 `.npz` committed before the dataset grew stay tracked on purpose: they
are a ready-made smoke-test dataset straight after a clone, and the ignore rule
only stops `git add .` from committing a 150,000-file build.

## Model

~43M parameters: sequence encoder 18.9M (6 blocks, `d=512`, 8 heads, RoPE),
map encoder 13.6M (4 blocks, plus a `1999 → 512` seed projection), expanders 10.5M
(`512 → 2048`, one per branch).

The map encoder seeds each residue with its own **contact map row**, indexed by
relative offset `j − i` and projected to `d=512` — not with a summary statistic.
Indexing by relative offset rather than absolute partner index is load-bearing:
see `SEED_MODE` in [map_encoder.py](map_encoder.py), and check (g) of its sanity
check, which measures the difference.
The expanders are training-time scaffolding — for downstream use
you want the sequence encoder:

```python
from train import load_sequence_encoder
enc = load_sequence_encoder()                 # defaults to $CKPT_DIR/best.pt
seq_repr, mask = enc(padded_ints, mask)       # (B, L, 512) per-residue reps
```
