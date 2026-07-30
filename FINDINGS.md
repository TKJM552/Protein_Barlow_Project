# Findings — Protein Barlow

Live lab notebook for the current architecture. Its predecessor's results are in
[FINDINGS_2026-07-26_jepa.md](FINDINGS_2026-07-26_jepa.md) and **do not carry
over** — that run had a predictor head and a 2-scalar map seed, both since
removed.

Written down because none of it is derivable from the code or git history.

---

## Status: nothing has been trained yet

The current architecture has **never completed a training run.** Every number
below comes from static analysis or an untrained model. Do not quote any of it as
a result.

| | |
|---|---|
| Architecture | 2 encoders + Barlow Twins, no prediction head |
| Parameters | 43.1M — sequence encoder 18.9M, map encoder 13.6M, expanders 10.5M |
| Training set | 4,719 chains of 4,966 (`MIN_RESIDUES = 40` drops 247) |
| Batching | length-bucketed, ≤4096 residues/batch, 286 steps/epoch |
| Last full run | none (the 50-epoch run in the archive used a different model) |
| Checkpoints | none valid; `checkpoints/best.pt` predates both changes |

---

## What changed, and why

**1. The map seed is now the contact row, indexed by relative offset.**
The old seed gave each residue two scalars (local degree, long-range degree),
which the archived run diagnosed as its ceiling: a plain `Linear` mapped every
residue into a rank-2 subspace, so the structure branch was nearly homogeneous
and matching it taught the sequence branch little. Measured on one 581-residue
chain:

| seed | width | mean pairwise cosine | rank |
|---|---|---|---|
| old 2-scalar | 2 | 0.906 | **2** |
| contact row, absolute index | 581 | 0.017 | 578 |
| contact row, relative offset | 1161 | 0.586 | **327** |

**2. No predictor.** The sequence branch previously ran through a per-residue MLP
before the loss. Since the Barlow Twins expanders are *already* a per-residue MLP
on each branch, that head added nothing they could not express. Removing it means
`sequence_encoder`'s own output is what the objective shapes — which is also the
tensor `load_sequence_encoder()` returns, so the thing being optimised and the
thing used downstream are now the same tensor.

**Unknown:** whether removing it helps, hurts, or does nothing. It reduces
sequence-branch capacity by 1.1M params and forces the encoder itself to be
alignable to the map branch. That could push more structure into the encoder or
over-specialise it. Nothing here measures which.

---

## This loss does not fit in fp16

**`AMP_DTYPE` defaulted to `fp16`, and this objective cannot be computed in it.**
Found by static analysis and confirmed against the archived run's own logs; no new
training involved.

`off_diag` is a sum of `D² − D = 4.19M` squared correlations. fp16 tops out at
65,504. The archived 50-epoch run's epoch-average `off_diag`:

| epoch | train | val |
|---|---|---|
| 1 | 51,852 | **81,135** |
| 2 | 84,303 | **89,642** |
| 50 | 12,049 | 12,640 |

Reproducing those magnitudes as an actual fp16 reduction: 51,852 → `51840.0`,
but 81,135 → `inf`. Under autocast the whole loss was computed at the autocast
dtype, so an fp16 run returns `inf` for the first several epochs, `train.py`'s
non-finite guard skips **every** optimizer step, and the run reports nothing but
`WARNING: frequent skips!`. Loss scaling cannot help — this is a loss *value*
overflow, not a gradient-scale one.

The archived run survived only because POD_SETUP said to pass `--amp-dtype bf16`.
The same file told you to *omit* it on V100/T4, which is the broken path.

`barlow_twins_core` now disables autocast and computes in fp32 (~15% of forward
FLOPs, the standard choice in reference BT/VICReg implementations). Check (d) of
`python barlow_twins.py` is the regression test. **Watch `off_diag` in epoch 1–3
of any future run** — it peaks there, and it is the number that decides whether a
precision change is safe.

---

## The trap that shaped the seed design

**A richer structure branch can be a POSITION LEAK, and every agreement metric in
`eval.py` would applaud it.** Found while building the row seed; no training
involved.

About 4 of every 10 contacts sit at `|i−j| ≤ 2`, present in 99.8% of those slots —
consecutive CA atoms are ~3.8 Å apart, always inside the 8 Å threshold. Index the
seed row by *absolute* partner index and that always-on band lands in columns
`i−2..i+2`: a pure function of `i`, identical in every protein. R² of predicting
from position alone, 120 chains:

| seeding | seed | z_map (random init) |
|---|---|---|
| absolute index | 0.573 | **0.620** |
| relative offset `j−i` | 0.031 | 0.033 |
| old 2-scalar | 0.047 | — |

62% of an absolute-indexed `z_map` is reproducible from RoPE position alone, so
the sequence branch could satisfy the objective while learning no structure — and
`on_diag`, CKA, and TEST 4 retrieval would all *improve*. **TEST 2 is the only
diagnostic that catches it**: a position-only solution survives rolling the maps,
so the real-vs-shuffled gap collapses.

Hence `SEED_MODE = "relative"`. Re-run check (g) of `python map_encoder.py` after
any change to how the map is fed in.

---

## What the untrained model already does

Cross-protein CKA between `z_map` vectors, untrained, first 120 residues:

```
        101m    103l    104m    107l    108m
101m   1.000   0.152   0.884   0.164   0.981
103l   0.152   1.000   0.166   0.866   0.164
104m   0.884   0.166   1.000   0.185   0.893
```

101m/104m/108m are myoglobins, 103l/107l are lysozymes. The map encoder groups by
fold **before any training** — architecture, not learning. This is exactly why
every metric needs an untrained baseline: a trained CKA of 0.88 between two
myoglobins would mean nothing on its own.

---

## Half the training compute was padding

Chains run 40–990 residues (median 218), and `collate_pad` pads every protein in a
batch up to its longest member. Measured over the whole dataset, random batches of
16 did **2.41×** the useful token work and **4.46×** on the L²-scaling terms — 52%
of an epoch's FLOPs went into padding.

Batching is now length-bucketed under a residue budget (`--residues-per-batch`,
default 4096). Padding drops to 1.00×, and worst-case `B × L²` falls 4.6×
(15.7M → 3.4M), which is what actually caused OOM.

The budget is in **residues, not proteins**, and that choice is load-bearing:
Barlow Twins treats each residue as a sample, so `N` per batch is a statistical
parameter, not just a memory knob. Length-bucketing at a fixed 16 proteins/batch
would have put 22% of batches below `EXPANDER_DIM = 2048` (as low as `N = 643`)
and swung the `off_diag` floor 21× across batches. The residue budget instead
holds `N` near-constant (median 3877, 0.3% of batches under 2048) — steadier than
the random batching it replaced.

**Unknown:** whether the changed batch composition affects the result. Batches are
now length-correlated, and 286 steps/epoch replaces 279, so the LR schedule
carries over — but nothing here measures whether bucketing changes what is learned.

---

## Next, in order

1. **Train it.** 50 epochs, `--amp-dtype bf16`, `--num-workers 8`. Nothing below
   is worth doing first, and nothing above is confirmed until this runs.
2. **TEST 2 immediately after.** It is the only guard against the position
   shortcut, and the seeding change is exactly what makes that risk live. A
   collapsed real-vs-shuffled gap invalidates every other number.
3. **Judge on `on_diag` and P@L/5**, never total loss — total falls forever via
   the off-diagonal term. Early-stop on `on_diag`.
4. **The scratch control.** A contact head plus the same model trained from
   random init; **the delta is the result**. Beating random init is a low bar and
   nothing in either findings file substitutes for this.
5. **Homology-aware split.** `random_split` puts homologous proteins in both
   train and val (see the myoglobins above), so held-out numbers are optimistic.
   Any supervised claim needs a sequence-identity or CATH-family split first.
6. **More data.** `get_files.py` caps the RCSB query at 5000 rows; the PDB has
   ~200k.

---

## Measurement traps

All of the traps in the [archived findings](FINDINGS_2026-07-26_jepa.md#measurement-traps)
still apply — they are properties of the metrics, not of the architecture. In
particular: never rank CKA across different sample counts, never mean-pool before
comparing, never compare the two branches by cosine, and give every metric an
untrained baseline.
