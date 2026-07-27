# Running this on a RunPod GPU

The sequence that actually worked, including the two things that cost the most
time on the first attempt.

## The one non-obvious thing

**Do not train out of `/workspace`.** RunPod's persistent storage is
network-backed. Constructing the dataset from it — 4,966 small `.npz` files —
took **over 5 minutes**, versus ~1 second on local disk. Training re-reads those
files every epoch, so it throttles the whole run, not just startup.

Clone to `/root` (local NVMe) and write only checkpoints to `/workspace`.
`/root` is wiped on pod restart, but re-cloning takes 30 seconds.

## Setup

Deploy with a **PyTorch template**, ~30 GB persistent storage (checkpoints are
~500 MB each; `--ckpt-every 5` over 50 epochs is ~5.5 GB). Then, in the web
terminal:

```bash
cd /root
git clone https://github.com/TKJM552/Protein_JEPA_Project.git
cd /root/Protein_JEPA_Project
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

Confirm the data loads fast (this is the check that catches the `/workspace` trap):

```bash
time python -c "import train,seq_encoder; seq_encoder.ProteinSequenceDataset(train.DATA_DIR)"
```

`real` should be 1–3 seconds. Minutes means you are reading from network storage.

## Train

```bash
python train.py --smoke-test        # one train + one val step, asserts grads reach all 4 groups
nohup python train.py --num-workers 8 --amp-dtype bf16 > train.log 2>&1 &
tail -f train.log
```

`nohup ... &` is what keeps training alive when the browser tab closes. Without
it, closing the terminal kills the run. `Ctrl+C` stops the `tail`, not the training.

- `--amp-dtype bf16` on Ampere or newer (A100, A10, L4, H100, 30xx/40xx/50xx).
  Omit it on V100/T4 to use the fp16 default.
- If you hit CUDA OOM, add `--batch-size 8`. Attention memory scales with
  `batch x length²` and chains run to ~990 residues. Barlow Twins treats each
  *residue* as a sample, so batch 8 still yields thousands — the statistics stay sound.

Checking back after a disconnect:

```bash
pgrep -af train.py                              # 1 trainer + 16 workers = healthy
tail -20 /root/Protein_JEPA_Project/train.log
nvidia-smi                                      # GPU-Util should be high
```

Watch for `WARNING: frequent skips!` — that means non-finite losses.

## Before terminating

**Terminating destroys `/workspace` too.** Get the checkpoint off the pod first:

```bash
runpodctl send /workspace/checkpoints/best.pt
runpodctl send /root/Protein_JEPA_Project/train.log
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

## Reference: timings on an RTX 4090

- ~1 min/epoch (280 steps at batch 16), 50 epochs in under an hour
- Dataset construction: ~1 s local disk, **5+ min on `/workspace`**
- Checkpoints ~493 MB each (43.1M params plus optimizer state)

## Private repo

If you set the repo back to private, the pod needs a token to clone:

```bash
git clone https://<token>@github.com/TKJM552/Protein_JEPA_Project.git
git remote set-url origin https://github.com/TKJM552/Protein_JEPA_Project.git  # scrub it
```

Use a **classic** token with the top-level `repo` scope
([github.com/settings/tokens](https://github.com/settings/tokens)). Fine-grained
tokens grant no repository access by default and fail with a misleading
"Write access to repository not granted" on a read-only clone.
