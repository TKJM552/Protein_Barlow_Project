# Protein JEPA

A joint-embedding predictive architecture (JEPA) for proteins, trained with a
Barlow Twins objective. A sequence branch reads amino acids; a structure branch
reads the CA-CA contact map. The predictor learns to map the sequence
representation onto the structure representation, so the sequence encoder ends up
carrying structural information without ever seeing coordinates at inference.

```
DataLoader -> SequenceEncoder -> Predictor -> pred  \
                                                     Barlow Twins loss
              MapEncoder ------------------------> target /
```

This is **joint** training: one AdamW optimizer holds the sequence encoder, the
predictor, the map encoder, and the Barlow Twins expanders. There is no frozen
branch, no stop-gradient, and no EMA — collapse is prevented purely by the loss's
off-diagonal redundancy term.

---

## Start here

- **[FINDINGS.md](FINDINGS.md)** — results of the 50-epoch run, why it plateaued,
  the ranked next experiments, and the measurement traps. Read before running
  another experiment.
- **[POD_SETUP.md](POD_SETUP.md)** — the RunPod sequence that works, including
  the `/workspace` slowness that costs 5 minutes per invocation if ignored.

## Quickstart on a GPU pod

The processed dataset is committed, so there is nothing to download:

```bash
git clone <your-repo-url>
cd Protein_JEPA_Project

# Most GPU images already ship a CUDA torch. If yours doesn't:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. Confirm the wiring on the GPU (a few seconds, no training)
python train.py --smoke-test

# 2. Train
python train.py --num-workers 8 --amp-dtype bf16
```

`--smoke-test` runs one train step and one val step, asserts the loss is finite,
and asserts gradients reach all four module groups. Run it first — it catches a
broken environment in seconds instead of at epoch 3.

To train against data or checkpoints living outside the repo:

```bash
export DATA_DIR=/workspace/processed_dataset
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
| `BATCH_SIZE` | `--batch-size` | `16` | Proteins per batch |
| `AMP_DTYPE` | `--amp-dtype` | `fp16` | `fp16` or `bf16` |

Training hyperparameters also have flags: `--epochs`, `--lr`, `--weight-decay`,
`--warmup-steps`, `--seed`, `--ckpt-every`, `--resume`, `--no-amp`.
Run `python train.py --help` for the full list.

### GPU notes

- **`--num-workers` matters.** The default of `0` decodes `.npz` files in the main
  process, which leaves the GPU idle between batches. Start at 8.
- **Prefer `--amp-dtype bf16` on Ampere or newer** (A100, A10, L4, H100, 30xx/40xx).
  bf16 has fp32's exponent range, so it needs no loss scaling and cannot overflow
  the way fp16 can. The `GradScaler` is enabled only for fp16.
- **Attention memory scales as `batch × length²`.** Chains run up to 1000 residues,
  so a batch that happens to contain several long ones is the worst case. If you
  hit OOM, lower `--batch-size` — the loss statistics stay sound as long as the
  batch still yields plenty of residues (Barlow Twins treats each *residue* as a
  sample, so batch 8 still gives thousands).
- **Checkpoints are ~500 MB each** (43M params × model + optimizer + scheduler
  state). With the default `--ckpt-every 5` over 50 epochs that is ~5.5 GB. Size
  your pod volume accordingly, or raise `--ckpt-every`.

### Resuming

```bash
python train.py --resume $CKPT_DIR/best.pt
```

Restores module weights, optimizer, scheduler, and scaler state, then continues
from the next epoch.

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
| `retrieval` | Is each protein's prediction closest to *its own* target? |
| `probe` | Does a frozen encoder + linear probe beat random init and a distance-only baseline on long-range contacts? |
| `alignment` | Do the two branches share a geometry? (CKA vs an untrained model) |

Without `--ckpt` the model is randomly initialized, which is a meaningful run —
it should produce the "broken/near-chance" numbers and confirm the diagnostics are
calibrated. `probe` is the load-bearing one: it must beat **both** random init and
the distance-only baseline, or the encoder only learned "near in sequence = near in
space."

---

## Rebuilding the dataset

Not needed for training — `processed_dataset/` is committed. Only do this to
change the contact threshold, the length cap, or the number of structures.

```bash
pip install -r requirements-data.txt
python get_files.py           # ~3.9 GB of .cif from RCSB -> $PDB_DIR
python get_inputs_outputs.py  # -> $DATA_DIR as one .npz per structure
```

Each `.npz` holds `sequence` (string), `seq_ints` (`int64`, `1..20`, `0` = pad),
and `contact_map` (`int8`, `(L, L)`, symmetric, 1-diagonal, CA-CA ≤ 8 Å).
Chains over 1000 residues are dropped.

Note that the RCSB query in [get_files.py](get_files.py) is not pinned to a
snapshot, so a rebuild today may not return exactly the same 5000 structures.

---

## Repository layout

| File | Role |
|---|---|
| [config.py](config.py) | Paths, device, workers, AMP — all env-overridable |
| [seq_encoder.py](seq_encoder.py) | Dataset, padding collate, token embedding, RoPE transformer encoder |
| [map_encoder.py](map_encoder.py) | Contact-map encoder; attention restricted to actual contacts |
| [predictor.py](predictor.py) | Per-residue MLP head mapping sequence reps → map reps |
| [barlow_twins.py](barlow_twins.py) | Expanders + the redundancy-reduction loss |
| [train.py](train.py) | Joint training loop, checkpointing, CLI |
| [eval.py](eval.py) | The six diagnostics above |
| [compare_embeddings.py](compare_embeddings.py) | One protein through both branches: CKA, per-residue cosine, ranking vs controls |
| [test_novel.py](test_novel.py) | Fetches unseen structures from RCSB and runs the same comparison |
| [get_files.py](get_files.py) | RCSB query + `.cif` download |
| [get_inputs_outputs.py](get_inputs_outputs.py) | `.cif` → sequence + contact map `.npz` |

Every module has a `__main__` sanity check; run any file directly to exercise it:

```bash
python seq_encoder.py    # shapes, symmetry, zero-padding, no padding leak
python map_encoder.py    # shapes, blindness to sequence, no padding leak
python predictor.py      # output shape, no NaNs
python barlow_twins.py   # grads on both sides, identity ≈ 0, collapse punished
```

### Not in git

`pdb_dataset/` (3.9 GB of raw `.cif`), `checkpoints/` (~500 MB per file, over
GitHub's 100 MB limit), and `.venv/`. See [.gitignore](.gitignore).

## Model

~43M parameters: sequence encoder 18.9M (6 blocks, `d=512`, 8 heads, RoPE),
map encoder 12.6M (4 blocks), predictor 1.1M, expanders 10.5M (`512 → 2048`).
The expanders and the predictor are training-time scaffolding — for downstream use
you want the sequence encoder:

```python
from train import load_sequence_encoder
enc = load_sequence_encoder()                 # defaults to $CKPT_DIR/best.pt
seq_repr, mask = enc(padded_ints, mask)       # (B, L, 512) per-residue reps
```
