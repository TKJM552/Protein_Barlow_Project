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
`pip install -r requirements.txt`** — if the template ships torch older than the
`>=2.2` floor, pip will upgrade it and can replace the working CUDA build with a
CPU-only wheel. Training needs only torch and numpy, both already present on any
PyTorch template.

## Build the dataset

The 150,000-structure dataset is not in git (only `pdb_ids.txt`, the 154,500 IDs
it is built from). Build it onto **local disk**:

```bash
pip install -r requirements-data.txt        # biopython, requests, scipy
export DATA_DIR=/root/processed_dataset     # NOT /workspace -- see above

nohup python get_files.py --build --workers 16 > build.log 2>&1 &
tail -f build.log
```

Expect **2–5 hours**. It streams gzipped mmCIF from RCSB, writes one `.npz` per
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
