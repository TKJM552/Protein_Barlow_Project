# Running this on a RunPod GPU

The sequence that actually worked, including the two things that cost the most
time on the first attempt.

## The one non-obvious thing

**Do not put the dataset on `/workspace`.** RunPod's persistent storage is
network-backed. Constructing the dataset from it — then 4,966 small `.npz` files —
took **over 5 minutes**, versus ~1 second on local disk. Training re-reads those
files every epoch, so it throttles the whole run, not just startup.

At **150,000 files** that same 300× penalty turns a ~40-second startup scan into
roughly **2.5 hours, every single invocation**. This is no longer a nuisance; it
makes the dataset size unusable. Build to `/root` (local NVMe) and write only
checkpoints to `/workspace`. `/root` is wiped on pod restart, but re-cloning takes
30 seconds and rebuilding is resumable.

## Setup

Deploy with a **PyTorch template**, ~30 GB persistent storage (checkpoints are
~500 MB each; the default run keeps `best.pt` plus a rolling `last.pt`, so ~1 GB).
Container disk needs ~5 GB free for the dataset (~0.5 GB) plus the image.
Then, in the web terminal:

> **Repo name.** The project is now *Protein Barlow*, but the GitHub remote is
> still `Protein_JEPA_Project` until it is renamed in the repo settings. The clone
> below therefore uses the old URL and names the local directory explicitly. After
> renaming on GitHub, swap the URL — GitHub redirects the old one, so both work.

```bash
cd /root
git clone https://github.com/TKJM552/Protein_JEPA_Project.git Protein_Barlow_Project
cd /root/Protein_Barlow_Project
export CKPT_DIR=/workspace/checkpoints
```

Verify the GPU before anything else:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True`. **If torch already imports, do not run
`pip install -r requirements.txt`** — pip may upgrade it and replace the working
CUDA build with a CPU-only wheel. Training needs only torch and numpy, both
already present on any PyTorch template.

Check the version by hand, though, because the `>=2.3` floor is real: `train.py`
builds its scaler with `torch.amp.GradScaler(device=...)`, which does not exist
in 2.2. The run would die at startup, *after* the slow dataset scan.

```bash
# Compares numerically, on purpose: a string compare puts '2.10' BELOW '2.3' and
# would send you into a needless `pip install -U torch`, which is the command most
# likely to swap the CUDA build for a CPU-only wheel. Also strips the +cu121 suffix.
python -c "import torch; v=tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])); assert v >= (2,3), torch.__version__; print('torch', torch.__version__, 'OK')"
```

If the template is older, upgrade from the CUDA index so you keep a GPU wheel —
never a bare `pip install -U torch`:

```bash
pip install -U torch --index-url https://download.pytorch.org/whl/cu121   # match nvidia-smi
```

## Build the dataset

The 150,000-structure dataset is not in git (only `pdb_ids.txt`, the 154,500 IDs
it is built from). Build it onto **local disk**:

```bash
pip install -r requirements-data.txt        # biopython, requests, scipy
export DATA_DIR=/root/processed_dataset     # NOT /workspace -- see above

nohup python get_files.py --build --workers 16 --max-per-cluster 5 > build.log 2>&1 &
tail -f build.log
```

**Use `--max-per-cluster 5`.** The 154,500-ID list is **7.2× redundant**: it
covers only **21,561 distinct proteins** at 30% sequence identity, and the top
1,000 clusters supply 48.5% of all structures while 7,986 clusters supply exactly
one. Building all of it downloads the same protein up to a thousand times and
then trains on that imbalance.

| cap | structures built | clusters kept |
|---|---|---|
| none | 154,463 | 21,561 |
| **5** | **59,161** | **21,561** |
| 10 | 79,369 | 21,561 |

A cap of 5 keeps **every** cluster at 2.6× less build time and 2.6× faster
epochs. It needs `pdb_clusters.txt`, which is committed — regenerate with
`python get_files.py --clusters` only if you want a different identity threshold.

Uncapped, expect **2–5 hours**. It streams gzipped mmCIF from RCSB, writes one `.npz` per
structure, and keeps no `.cif` — ~0.5 GB lands on disk instead of ~118 GB, so the
30 GB pod is fine. Progress lines report `built / target`, rate, and an ETA.

It is safe to interrupt. Every `.npz` is written atomically (temp file + rename),
and a re-run skips IDs that already have one, so the same command resumes. Sanity
checks worth 20 seconds before walking away:

```bash
head -3 build.log                            # should say "0 structures already in ..."
ls /root/processed_dataset | wc -l            # climbing
du -sh /root/processed_dataset                # ~3 KB per structure
ls /root/Protein_Barlow_Project/pdb_dataset 2>/dev/null | wc -l   # must stay 0/absent
```

That last one is the check that the streaming path is really streaming: if
`pdb_dataset/` starts filling up, you ran `--download-cif` by mistake and are
heading for ~118 GB.

Then confirm the data loads fast (this is the check that catches the `/workspace`
trap):

```bash
time python -c "import train,seq_encoder; seq_encoder.ProteinSequenceDataset(train.DATA_DIR)"
```

`real` should be **~40 seconds** for 150,000 files on local disk. Several minutes
or more means you are reading from network storage — move the dataset to `/root`.

## Train

```bash
python train.py --smoke-test        # one train + one val step, asserts grads reach all 3 groups
nohup python train.py --num-workers 8 --amp-dtype bf16 --epochs 5 > train.log 2>&1 &
tail -f train.log
```

**`--epochs` matters now.** `EPOCHS = 50` was tuned when an epoch was 286 steps.
At 150,000 structures an epoch is ~9,640 steps (~35 min on a 4090), so the default
is ~480,000 steps and about a day of GPU time. `--epochs 5` is already 2.4× the
total gradient steps of the archived 50-epoch run — start there and read the val
curve. Confirm the banner before it commits: it prints `steps/epoch` and
`total steps` up front.

`nohup ... &` is what keeps training alive when the browser tab closes. Without
it, closing the terminal kills the run. `Ctrl+C` stops the `tail`, not the training.

- `--amp-dtype bf16` on Ampere or newer (A100, A10, L4, H100, 30xx/40xx/50xx).
  Omit it on V100/T4 to use the fp16 default — that is now safe. The Barlow Twins
  loss is computed in fp32 regardless of `--amp-dtype`, because `off_diag` runs to
  ~90,000 in the first epochs and fp16 caps at 65,504. Before that fix an fp16 run
  produced `inf`, skipped every optimizer step, and reported only
  `WARNING: frequent skips!`.
- If you hit CUDA OOM, add `--residues-per-batch 2048` (**not** `--batch-size`,
  which only affects `eval.py` now). Batches are length-bucketed and close at
  `proteins x longest chain <= the budget`, so that number is the padded width
  directly — halving it halves peak activation memory. Stay above ~2048: Barlow
  Twins treats each residue as a sample and wants more of them than
  `EXPANDER_DIM = 2048` per batch.

Checking back after a disconnect:

```bash
pgrep -af train.py                              # 1 trainer + 16 workers = healthy
tail -20 /root/Protein_Barlow_Project/train.log
nvidia-smi                                      # GPU-Util should be high
```

Watch for `WARNING: frequent skips!` — that means non-finite losses.

## The `free` column

Every epoch line ends with `free NNN`: the fraction of `z_seq` reproducible from
**position and chain length alone** — everything the model got without reading
any chemistry. **High is bad.**

It exists because Barlow Twins treats each residue as a sample, and both branches
can work out where they are in the chain. `z_seq = z_map = f(position)` satisfies
the objective perfectly while learning nothing, it is an attractor the model can
drift into at *any* epoch, and **val loss falls as it drifts** — so val loss will
never warn you, and `best.pt` is still selected on val loss. This column is the
only thing watching.

Reference points, measured:

| | `free` |
|---|---|
| random noise (the estimator's own floor) | 0.005 |
| **untrained `z_seq` — where your run starts** | **0.018** |
| a purely positional representation | 0.994 |

So anything up around 0.1+ and climbing means the shortcut is forming: kill the
run rather than pay for the remaining epochs. Read the trend, not one value — it
is biased upward by about (cells / residues) and is not comparable across
different bin counts.

## The train/val split is grouped by sequence cluster

The banner prints `split by sequence cluster: N clusters -> ... (no cluster spans
both)`. If instead you see a **WARNING** about a random split, `pdb_clusters.txt`
is missing — run `python get_files.py --clusters`.

This matters more than it sounds. The PDB deposits the same protein repeatedly as
mutants, ligand complexes and better resolutions, so a random split puts identical
chains on both sides. Measured on the committed 4,966-structure dataset:

| split | val chains with an exact sequence twin in train | with a near-twin |
|---|---|---|
| random | **40.3%** | 73.9% |
| grouped by 30% identity | **0.8%** | 4.4% |

Two in five validation proteins were literally in the training set, so val loss
was substantially measuring memorisation — and `best.pt` is selected on val loss.

Because of this, **val losses from before this change are not comparable** to
ones after it; the old numbers are optimistic. Checkpoints record which split
they used under `split.grouped_by_cluster`.

## Baseline first, fix only if needed

**The positional shortcut is argued for but has never been observed in this
model.** Untrained `z_seq` sits at `free = 0.018` against a 0.005 noise
floor, and no run with the current architecture has finished. So the default run
applies **no fix** — it is the measurement.

The fix (`--position-centering`) subtracts the per-index mean before the loss,
which makes a purely positional representation worth exactly zero — verified, it
centres to 0.0000. But it is not free: it also removes real population-level
chemistry, since terminal charge and end-of-chain disorder are shared by every
protein at that index. Turning it on by default would mean never learning whether
it was needed.

**Step 1 — run the baseline and watch `free`.** Do it on the committed
4,966-structure dataset first: no build required, and a training step costs the
same whatever the dataset size, so nothing is saved by waiting for 150k.

```bash
unset DATA_DIR                      # use the committed processed_dataset/
python train.py --epochs 10 --seed 0 --ckpt-dir ./ck_base
```

**Step 2 — read the `free` column, from epoch 1.** You do not need the run to
finish:

- stays around **0.02**, where it starts → no shortcut. The fix was never needed and this run is
  your model. Go straight to the 150k build.
- climbs past **~0.05** and keeps rising → the shortcut is forming. Kill the run
  at epoch 3–5 rather than paying for the rest.

A single elevated value is not the signal; the trend across epochs is.

**Step 3 — only if it climbed**, re-run with the fix and compare:

```bash
python train.py --epochs 10 --seed 0 --ckpt-dir ./ck_fix --position-centering
python eval.py --test probe --ckpt ./ck_base/best.pt     # and ck_fix
```

Same `--seed`, so the split and batch order are identical and the only difference
is the change. TEST 5 (`probe`) is the arbiter — it asks whether the frozen
encoder still contains contact information, which is the direct test of "did the
fix cost me anything real". If `probe` holds and `free` drops, take the fix to
the 150k run.

Two caveats if you get that far. 4,966 structures is not 150,000, so this screens
the *mechanism*, not the final numbers — and keep watching `free` during the real
run, since shortcut emergence is the part that genuinely does not transfer. And
`eval.py` does not itself centre: with centering on, the component of `z` along
the per-index mean gets no gradient, so it is unconstrained noise in the tests
that mix indices (`retrieval`, `alignment`). That biases the comparison
**against** the fix, so a win for it is trustworthy; a loss on those two
specifically is the confound, not a finding, and `probe` is the tiebreak.

## Before terminating

**Terminating destroys `/workspace` too.** Get the checkpoint off the pod first:

```bash
runpodctl send /workspace/checkpoints/best.pt
runpodctl send /root/Protein_Barlow_Project/train.log
```

Each prints a one-time code; run the matching `runpodctl receive <code>` on your
own machine. On macOS without Homebrew:

```bash
mkdir -p ~/bin && curl -sL -o ~/bin/runpodctl \
  https://github.com/runpod/runpodctl/releases/download/v2.7.2/runpodctl-darwin-arm64
chmod +x ~/bin/runpodctl
```

Verify the checkpoint before terminating:

```bash
python -c "import torch; c=torch.load('best.pt',map_location='cpu',weights_only=False); print(c['epoch'], c['val_loss'])"
```

## Reference: timings

At 150,000 structures (extrapolated from a measured 1,067-structure random sample;
measure your first epoch rather than trusting these):

- Dataset build: **2–5 h** at `--workers 16`. 7.7 structures/s measured on a home
  connection at only 220% CPU of 8 cores, i.e. network-bound — a pod's network
  should do better, at which point pure-Python `MMCIFParser` (~59 ms/structure,
  2.5 CPU-hours total) becomes the limit and more workers help.
- Dataset scan at startup: **~40 s** local disk, ~2.5 h on `/workspace`
- Epoch on a 4090: **~35 min** (~9,640 steps at ≤4096 residues/batch). The
  archived 50-epoch run was ~1 min/epoch at 286 steps on 4,966 structures.
- Padding efficiency stays at 0.989 at this scale (length bucketing still works;
  batches hold 4–97 proteins, median 12)
- Checkpoints ~500 MB each (43.1M params plus optimizer state)
- Peak host RAM for the dataset object: ~220 MB at 150,000 files

## Private repo

If you set the repo back to private, the pod needs a token to clone:

```bash
git clone https://<token>@github.com/TKJM552/Protein_JEPA_Project.git Protein_Barlow_Project
cd Protein_Barlow_Project
git remote set-url origin https://github.com/TKJM552/Protein_JEPA_Project.git  # scrub it
```

Use a **classic** token with the top-level `repo` scope
([github.com/settings/tokens](https://github.com/settings/tokens)). Fine-grained
tokens grant no repository access by default and fail with a misleading
"Write access to repository not granted" on a read-only clone.
