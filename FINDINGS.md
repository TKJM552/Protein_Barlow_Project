# Findings — Protein Barlow

Live lab notebook for the current architecture. Its predecessor's results are in
[FINDINGS_2026-07-26_jepa.md](FINDINGS_2026-07-26_jepa.md) and **do not carry
over** — that run had a predictor head and a 2-scalar map seed, both since
removed.

Written down because none of it is derivable from the code or git history.

---

## Status: first run complete — it trains, and it takes the shortcut

| | |
|---|---|
| Architecture | 2 encoders + Barlow Twins, no prediction head |
| Parameters | 43.1M — sequence encoder 18.9M, map encoder 13.6M, expanders 10.5M |
| Training set | 4,246 train / 473 val, from **1,547 sequence clusters** |
| Batching | length-bucketed, ≤4096 residues/batch, 283 steps/epoch |
| First full run | 2026-08-03, 50 epochs, 4090, bf16, `--seed 0`, no positional fix |
| Best checkpoint | **epoch 17**, val 562.167 (val loss rises after that) |
| Centering arm | same seed/epochs with `--position-centering`; best **epoch 9**, val 1071.370 |
| Verdict | centering **prevents** the shortcut (`free` 0.172 → 0.065) at no measured cost |

---

## The first run: three findings

**1. It trains.** `on_diag` — which sits at ~2048 when the two branches agree on
nothing — fell to **13.1** on train. The sequence and contact-map views do learn
to agree. That was not established before this run.

**2. It overfits badly.** Train and validation diverge from about epoch 17:

| | epoch 17 | epoch 50 |
|---|---|---|
| train `on_diag` | 52 | **13** |
| val `on_diag` | 400 | **545** |

Best val loss is 562.167 at epoch 17; by epoch 50 it is back to 661. A 40×
train/val gap is memorisation, which is what a 43M-parameter model does to 1,547
distinct proteins. **This is the run's biggest problem, and it is a dataset-size
problem, not a loss problem** — scaling to the 21,561 clusters in `pdb_ids.txt`
is what addresses it. Nothing about the positional fix touches it.

**3. The positional shortcut is real, and it plateaus rather than running away.**
`free` (share of `z_seq` reproducible from position and chain length alone):

| epoch | 1 | 3 | 6 | 17 | 50 |
|---|---|---|---|---|---|
| `free` | 0.043 | 0.187 | **0.250** | 0.229 | **0.172** |

Against 0.018 untrained and a 0.005 noise floor. So it rises fast for ~6 epochs,
peaks at a quarter of the representation, then falls back and settles near a
sixth for the last ten epochs.

That shape is the finding. Position is picked up early as a scaffold and is
*partly* displaced as real chemistry is learned — but it stalls, and ~17% of the
representation is still obtainable without reading any chemistry at the end of
training. It is neither the total collapse the argument feared nor absent.

Note `best.pt` is chosen on val loss, so it is epoch 17, where `free` is **0.229**
— *higher* than the epoch-50 model. Selecting on val loss actively prefers the
more positional checkpoint.

### Reading the two shortcut metrics

Both are printed every epoch and both are also in `eval.py --test shortcut`.
They answer the same question by different routes, so agreement is worth more
than either alone:

| | 0.005 | 0.018 | 0.994 |
|---|---|---|---|
| `free` | estimator's noise floor | untrained encoder | purely positional |

`shuf` — how much of `z_seq` survives permuting the amino acids within a chain —
reads 0.009 untrained and 1.000 for a position-only encoder. Where they disagree,
trust `free`: a permuted chain is not a protein, so `shuf` asks the encoder about
an input unlike anything it trained on.

---

## The centering arm: it PREVENTS the shortcut, and costs nothing

Same seed, same 50 epochs, same data, `--position-centering` the only difference.

**1. `free` never rises.**

| epoch | 1 | 3 | 6 | 9 | 20 | 50 |
|---|---|---|---|---|---|---|
| baseline | 0.043 | 0.187 | **0.250** | 0.250 | 0.208 | 0.172 |
| centered | 0.045 | 0.059 | 0.067 | 0.063 | 0.073 | **0.065** |

Flat at 0.06–0.07 for all fifty epochs. The baseline's climb to a quarter of the
representation by epoch 6 simply does not happen. This is **prevention, not
suppression** — worth knowing, because it means the fix does not depend on the
shortcut forming first and being pushed back out.

**2. It costs nothing measurable.** TEST 5, long-range contacts, on each arm's
`best.pt`:

| | P@L/5 | AUC |
|---|---|---|
| baseline (epoch 17) | 0.050 | 0.609 |
| centered (epoch 9) | **0.053** | **0.611** |
| random encoder | 0.033 | 0.565 |
| distance-only | 0.026 | 0.760 |

Slightly *better*, not worse. The concern that removing the per-index mean would
take real chemistry with it is not supported: nothing the contact probe can see
was lost.

**3. It does not fix the overfitting**, and was never going to. At epoch 50 both
arms sit at a **42×** train/val `on_diag` gap. That is a dataset-size problem —
1,547 distinct proteins against 43M parameters — and it is what scaling to 21,561
clusters is for.

### DO NOT compare val loss between the arms

Baseline best val is 562.167; the centered arm's is 1071.370. **This does not mean
the fix made the model worse.**

The centered run computes its val loss *with centering applied*. Centering removes
the positional component of agreement, which mechanically lowers every `c_ii` and
so raises `on_diag`. It is a stricter objective by construction — a different
quantity, not a worse score on the same one.

The only cross-arm comparisons that mean anything are **`free`** and **TEST 5**,
because both are computed identically in both arms on the frozen encoder. Both
favour centering.

### What is still unexplained

- `free` settles at **0.065**, not at the 0.018 an untrained encoder gives. So
  centering does not drive positional content to zero. It removes position from
  the conditional MEAN — exactly and provably — and whatever remains is outside
  the mean, i.e. nonlinear or higher-moment. A trained adversarial probe is the
  only instrument that would resolve it; nothing here says it is worth building.
- The encoder is weak in absolute terms. **`distance-only` gets AUC 0.760 against
  the encoder's 0.611** — a trivial "near in sequence = near in space" prior still
  wins on global ranking. The encoder beats it on P@L/5 (0.053 vs 0.026), the
  field-standard metric, so "trained > distance-only" holds on one measure and
  fails on the other. It has learned something real and modest, not something
  strong.
- Best val arrives at **epoch 9** with centering versus **epoch 17** without, so
  the centered arm starts overfitting sooner. Unexplained; possibly just noise in
  a val set of 473 chains.

**Decision: run the 150k build with `--position-centering`.** It prevents the
shortcut at no measured cost, and the open question — whether the shortcut
re-emerges at 14× the diversity — is the one thing this dataset cannot answer.

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
