# Running this on a RunPod GPU

The sequence that actually worked, including the things that cost the most time
on the first attempt.

**The encoder run is DONE.** 150,169 structures, 40 epochs, centering on,
2026-08-05 — results in [FINDINGS.md](FINDINGS.md). Retrieval on unseen families
holds (54% top-1 vs 6% random); the linear contact probe does not (0.028 vs 0.029
random). Do not redo it, and do not redo the 4,966-structure baseline before it.

What is left to run on a pod is the **contact predictor**, in two arms — jump to
*The contact predictor* at the bottom. The encoder sections are kept because you
need them to rebuild the dataset, and because the traps in them are not specific
to which model you are training.

## The one non-obvious thing

**Do not put the dataset on `/workspace`.** RunPod's persistent storage is
network-backed. Constructing the dataset from it — then 4,966 small `.npz` files —
took **over 5 minutes**, versus ~1 second on local disk. Training re-reads those
files every epoch, so it throttles the whole run, not just startup.

At **154,500 files** that same 300× penalty turns a ~40-second startup scan into
hours, **every single invocation**. This is no longer a nuisance; it makes the
dataset size unusable. Build to `/root` (local NVMe) and write only checkpoints to
`/workspace`. `/root` is wiped on pod restart, but re-cloning takes 30 seconds and
rebuilding is resumable.

## Setup

Deploy an **RTX 4090** with a **PyTorch template**, ~30 GB persistent storage and
~20 GB container disk.

Storage is not the constraint. The builder streams gzipped mmCIF and keeps only a
small `.npz` per structure, so the complete 154,500-structure dataset is **0.43 GB**
— it never touches the ~118 GB the raw `.cif` files would take. The 30 GB volume is
for checkpoints, which are ~500 MB each (the run keeps `best.pt` plus a rolling
`last.pt`).

Everything below runs in RunPod's **web terminal** (Connect → Start Web Terminal).

```bash
cd /root
git clone https://github.com/TKJM552/Protein_Barlow_Project.git
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

The dataset is not in git (only `pdb_ids.txt`, the 154,500 IDs it is built from).
Build it onto **local disk**:

```bash
pip install -r requirements-data.txt        # biopython, requests, scipy
export DATA_DIR=/root/processed_dataset     # NOT /workspace -- see above

nohup python get_files.py --build --workers 16 > build.log 2>&1 &
tail -f build.log
```

No `--target` and no `--max-per-cluster`: the defaults already aim at the full
150,000, and this run is deliberately uncapped. `Ctrl+C` stops the `tail`, not the
build.

Expect **2–5 hours**. Progress lines report `built / target`, rate, and an ETA.

### Why uncapped, and what capping would have bought

`--max-per-cluster N` keeps at most N structures per 30%-identity sequence cluster.
It is worth understanding what it does and does not change:

| cap | structures built | **clusters kept** | disk | steps/epoch |
|---|---|---|---|---|
| 1 | 21,598 | 21,598 | 0.06 G | ~1,230 |
| 5 | 59,198 | 21,598 | 0.17 G | ~3,373 |
| 10 | 79,406 | 21,598 | 0.22 G | ~4,525 |
| 20 | 98,831 | 21,598 | 0.28 G | ~5,632 |
| **none** | **154,500** | **21,598** | **0.43 G** | **~8,800** |

**Every cap yields the same 21,598 distinct proteins.** The ID list is 7.2×
redundant — the PDB deposits the same protein repeatedly as point mutants, ligand
complexes, alternative conformations and better resolutions — so the cap only
changes how many near-duplicates of each family you get, never how many families.

Two real considerations, pulling opposite ways:

- **Against uncapped:** the redundancy is lopsided. The top 1,000 clusters supply
  **48.5%** of all structures while **8,023** clusters supply exactly one, so
  uncapped the model spends about half its time on 5% of the families.
- **For uncapped:** those duplicates are not identical. Different conformations and
  crystal forms of the same protein give genuinely different contact maps, which is
  natural augmentation working *against* memorisation — and memorisation is this
  model's measured failure (a 42× train/val gap, see FINDINGS).

Compute is **not** a reason to cap, as long as you compare at fixed gradient steps
rather than fixed epochs. At ~132,000 steps, uncapped gets 15 passes over 154,500
structures and `--max-per-cluster 5` gets 40 passes over 59,198 — same GPU hours,
and more distinct data is the better shape for an overfitting problem. Disk is
irrelevant at every setting. The only genuine extra cost of uncapped is a longer
build (2–5 h rather than 1–2 h).

Capping needs `pdb_clusters.txt`, which is committed; regenerate with
`python get_files.py --clusters` only for a different identity threshold.

### Checks while it runs

It is safe to interrupt. Every `.npz` is written atomically (temp file + rename),
and a re-run skips IDs that already have one, so the same command resumes.

```bash
head -3 build.log                            # should say "0 structures already in ..."
ls /root/processed_dataset | wc -l           # climbing
du -sh /root/processed_dataset               # ~3 KB per structure, heading for ~0.43 G
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

`real` should be **~40 seconds** for 154,500 files on local disk. Several minutes
or more means you are reading from network storage — move the dataset to `/root`.

## Train

```bash
python train.py --smoke-test        # one train + one val step, asserts grads reach all 3 groups

nohup python -u train.py --amp-dtype bf16 --num-workers 8 --seed 0 \
      --position-centering --epochs 40 > train.log 2>&1 &
tail -f train.log
```

`nohup ... &` is what keeps training alive when the browser tab closes. Without
it, closing the terminal kills the run. `Ctrl+C` stops the `tail`, not the training.

**`python -u` is not optional.** With stdout redirected to a file Python buffers
in 8 KB blocks, so the ~1.5 KB banner sits unwritten for minutes and the log looks
empty — and a hard kill loses whatever is still in the buffer, which is exactly
the output you would want.

Before it commits, the banner prints `steps/epoch` and `total steps`. Measured on
the real run: **9,756 steps/epoch**, 390,240 total, **~9 min/epoch**, ~6 h for 40
epochs. (An earlier version of this file said 32 min/epoch. That was extrapolated
from the 4,966-structure run's 1 min/epoch by step count, which wrongly scaled the
fixed per-epoch costs — the val pass, `free`, the probe — that were most of that
minute. Measure your first epoch; do not extrapolate.)

It also prints `split by sequence cluster: N clusters`; a **WARNING** about a
random split instead means `pdb_clusters.txt` is missing.

Why each flag:

- **`--position-centering` — on, not off.** This is the change from the earlier
  runbook. The shortcut was argued for, then measured: baseline `free` climbed
  0.043 → 0.250 by epoch 6, while the centered arm stayed flat at 0.06–0.07 for all
  fifty epochs, at no measured cost to contact prediction (P@L/5 0.050 → 0.053).
  The baseline question is answered; there is no reason to pay for it twice.
- **`--epochs 40`.** The cosine LR schedule anneals to zero at exactly this number,
  so it is locked in at launch. Resuming later with a different `--epochs` puts a
  discontinuity in the LR, and a `best.pt` from a longer schedule is un-annealed
  and slightly degraded — so overshooting is not free, and neither is stopping
  short. `EPOCHS = 50` in config was tuned when an epoch was 286 steps.
- **`--keep-epoch-ckpts --ckpt-every 1`** if you want to compare epochs later.
  `--keep-epoch-ckpts` alone does NOT save every epoch: it only changes whether the
  periodic save is kept, and the cadence is `CKPT_EVERY_EPOCHS = 5`. Without both
  flags `last.pt` is a rolling file, and the 40-epoch run's epoch-8 weights — the
  best on val agreement — were overwritten and are unrecoverable.
- **`--amp-dtype bf16`** on Ampere or newer (A100, A10, L4, H100, 30xx/40xx/50xx).
  Omit it on V100/T4 to use the fp16 default — that is now safe. The Barlow Twins
  loss is computed in fp32 regardless of `--amp-dtype`, because `off_diag` runs to
  ~90,000 in the first epochs and fp16 caps at 65,504. Before that fix an fp16 run
  produced `inf`, skipped every optimizer step, and reported only
  `WARNING: frequent skips!`.
- **`--seed 0`** pins the train/val split and the batch order.

If you hit CUDA OOM, add `--residues-per-batch 2048` (**not** `--batch-size`,
which only affects `eval.py` now). Batches are length-bucketed and close at
`proteins x longest chain <= the budget`, so that number is the padded width
directly — halving it halves peak activation memory. Stay above ~2048: Barlow
Twins treats each residue as a sample and wants more of them than
`EXPANDER_DIM = 2048` per batch.

Checking back after a disconnect:

```bash
pgrep -af train.py                              # 1 trainer + 8 workers = healthy
tail -20 /root/Protein_Barlow_Project/train.log
nvidia-smi                                      # GPU-Util should be high
```

Three things to watch in the epoch lines:

| column | healthy | bad |
|---|---|---|
| `free` | flat, ~0.02–0.07 | climbing past 0.1 — centering is not holding |
| val `on_diag` | falling | rising for 3+ epochs — past the useful point |
| `WARNING: frequent skips!` | absent | present — non-finite losses |

You do not have to stop the moment val turns. `best.pt` tracks the best val epoch
throughout, so overshooting costs GPU time and nothing else.

**Val loss from this run is not comparable to the archived baseline's numbers.**
With centering on, the model scores itself with the positional component of
agreement removed, which mechanically raises `on_diag`. It is a stricter objective,
not a worse model. See FINDINGS.

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
| untrained `z_seq` | 0.018 |
| **centered 50-epoch run, every epoch** | **0.06–0.07** |
| uncentered 50-epoch run, peak at epoch 6 | 0.250 |
| a purely positional representation | 0.994 |

With `--position-centering` on, anything climbing past ~0.1 means centering is not
containing it at this scale, which would itself be the finding — that is the open
question this run exists to answer. Read the trend, not one value: it is biased
upward by about (cells / residues) and is not comparable across different bin counts.

## The train/val split is grouped by sequence cluster

The banner prints `split by sequence cluster: N clusters -> ... (no cluster spans
both)`. If instead you see a **WARNING** about a random split, `pdb_clusters.txt`
is missing — run `python get_files.py --clusters`.

This matters more than it sounds. The PDB deposits the same protein repeatedly as
mutants, ligand complexes and better resolutions, so a random split puts identical
chains on both sides. Measured on the 4,966-structure dataset:

| split | val chains with an exact sequence twin in train | with a near-twin |
|---|---|---|
| random | **40.3%** | 73.9% |
| grouped by 30% identity | **0.8%** | 4.4% |

Two in five validation proteins were literally in the training set, so val loss
was substantially measuring memorisation — and `best.pt` is selected on val loss.
Uncapped, the redundancy is 7.2× rather than 3×, so this matters *more* here, not
less.

Because of this, **val losses from before this change are not comparable** to
ones after it; the old numbers are optimistic. Checkpoints record which split
they used under `split.grouped_by_cluster`.

## Evaluate before terminating

```bash
python eval.py --test probe    --ckpt /workspace/checkpoints/best.pt
python eval.py --test shortcut --ckpt /workspace/checkpoints/best.pt
python eval.py --test shuffled --ckpt /workspace/checkpoints/best.pt
```

`probe` (TEST 5) is the one that decides whether this run was worth it. The
comparison is against the centered 4,966-structure run: **P@L/5 0.053, AUC 0.611**.
This run's entire premise is that 31× more data closes the 42× train/val gap; if
P@L/5 has not moved, more data was not the answer.

`shortcut` (TEST 7) re-measures `free` on the frozen checkpoint and also reports
`shuf`, the share of `z_seq` that survives permuting the amino acids within a
chain. Note `best.pt` is selected on val loss, which in the archived baseline chose
the *more* positional checkpoint — so check `free` on the actual saved model, not
just the last epoch line.

## The contact predictor — the run that is actually outstanding

Two arms, identical in every way except what feeds the predictor. **The delta is
the result.** One arm alone answers nothing, so budget for both before starting
either.

```bash
python train_contact.py --arm scratch --smoke-test          # ~1 min, checks wiring

nohup python -u train_contact.py --arm scratch --seed 0 \
      --amp-dtype bf16 --num-workers 8 > scratch.log 2>&1 &

nohup python -u train_contact.py --arm pretrained --seed 0 \
      --amp-dtype bf16 --num-workers 8 \
      --encoder-ckpt /workspace/checkpoints/best.pt > pretrained.log 2>&1 &
```

Run them **sequentially**, not both at once — they will fight over the GPU and
neither timing will mean anything.

Each epoch line looks like:

```
epoch  1 | train 0.6210 | val 0.6483 | P@L/5 >12 0.081 (dist 0.034) | >=24 0.052 (dist 0.021)
```

Read it in this order, and do not skip the first:

1. **Does EITHER arm beat the `dist` column?** `dist` is the trivial "closer in
   the chain = more likely to touch" rule. FINDINGS measured it *beating* the
   pretrained encoder on AUC (0.758 vs 0.636), so an arm that does not clear it
   has learned the `|i-j|` prior and nothing else, and nothing below it matters.
2. **Does `pretrained` beat `scratch`?** That is the pretraining question, and the
   reason both files exist.
3. Absolute numbers, last.

`>12` is this repo's long-range cut; `>=24` is the literature's. Quote the second
if you compare to anyone else's number.

### Memory

Batches are sized by **pair count** (`B x L_max^2`), not residue count, because
that is what an L^2 model actually costs — 4 proteins of 1024 residues and 20 of
205 are both 4096 residues but 4.2M vs 0.84M pairs. So `--pair-budget` is the OOM
knob, and halving it is the first thing to try.

After the first step the run prints **measured** peak GPU memory. Retune
`--pair-budget` from that, not from the estimates in `contact_predictor.py` — those
are derived from reading the code and are expected to be off.

Do **not** reach for `--crop` first. It defaults to 0 (whole proteins) on purpose:
cropping means pairs further apart than the crop are never a training target,
which caps what the model can learn and not just what fits.

### After the arms

```bash
python train_contact.py --arm <winner> --no-relpos --seed 0 ...    # the control
```

`--no-relpos` removes the `|i-j|` embedding. If the score barely drops, the offset
table was doing the work rather than the sequence.

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
export PATH="$HOME/bin:$PATH"     # or call it as ~/bin/runpodctl
```

Verify the checkpoint before terminating:

```bash
python -c "import torch; c=torch.load('best.pt',map_location='cpu',weights_only=False); print(c['epoch'], c['val_loss'])"
```

## Reference: timings

At 154,500 structures (extrapolated from a measured 1,067-structure random sample
and the archived 4,966-structure run; measure your first epoch rather than trusting
these):

- Dataset build: **2–5 h** at `--workers 16`. 7.7 structures/s measured on a home
  connection at only 220% CPU of 8 cores, i.e. network-bound — a pod's network
  should do better, at which point pure-Python `MMCIFParser` (~59 ms/structure,
  2.5 CPU-hours total) becomes the limit and more workers help.
- Dataset scan at startup: **~40 s** local disk, hours on `/workspace`
- Encoder epoch on a 4090: **~9 min, measured** (9,756 steps at ≤4096
  residues/batch), so ~6 h for 40 epochs. The archived 50-epoch run was ~1
  min/epoch at 286 steps on 4,966 structures — **do not scale that by step count**,
  most of that minute was the fixed per-epoch validation, `free` and probe costs,
  and scaling it overshot by 4×.
- Dataset tarball: `tar -C /root -cf /workspace/processed_dataset.tar
  processed_dataset` is ~0.43 GB and saves the 2–5 h rebuild. Worth it for
  reproducibility too: a rebuild can differ by a handful of transient RCSB
  failures, which shifts the cluster split and makes future numbers not strictly
  comparable to the ones in FINDINGS.
- Padding efficiency stays at 0.989 at this scale (length bucketing still works;
  batches hold 4–97 proteins, median 12)
- Checkpoints ~500 MB each (43.1M params plus optimizer state)
- Peak host RAM for the dataset object: ~220 MB

## Private repo

If you set the repo back to private, the pod needs a token to clone:

```bash
git clone https://<token>@github.com/TKJM552/Protein_Barlow_Project.git
cd Protein_Barlow_Project
git remote set-url origin https://github.com/TKJM552/Protein_Barlow_Project.git  # scrub it
```

Use a **classic** token with the top-level `repo` scope
([github.com/settings/tokens](https://github.com/settings/tokens)). Fine-grained
tokens grant no repository access by default and fail with a misleading
"Write access to repository not granted" on a read-only clone.
