# Findings — Protein Barlow

Live lab notebook for the current architecture. Its predecessor's results are in
[FINDINGS_2026-07-26_jepa.md](FINDINGS_2026-07-26_jepa.md) and **do not carry
over** — that run had a predictor head and a 2-scalar map seed, both since
removed.

Written down because none of it is derivable from the code or git history.

---

## Status: it retrieves unseen proteins, and it does not predict contacts

The full-scale run is done. The one-line summary is that **the model learned
something real and the two headline diagnostics disagree about whether that
something is useful**, which is the state of play as of 2026-08-05.

| | |
|---|---|
| Architecture | 2 encoders + Barlow Twins, no prediction head |
| Parameters | 43.1M — sequence encoder 18.9M, map encoder 13.6M, expanders 10.5M |
| Current run | 2026-08-05, **150,169 structures**, 21,265 clusters, 40 epochs, 4090, bf16, `--position-centering`, `--seed 0` |
| Split | 135,147 train / 15,022 val, grouped by 30% identity |
| Cost | 9,756 steps/epoch, 390,240 steps total, ~6 h |
| Best checkpoint | **epoch 40**, val 596.319 — but see the selection bug below |
| **Holds** | cross-modal retrieval on unseen families: **54% top-1 vs 6% random**, and rising with training |
| **Fails** | linear contact probe: **P@L/5 0.028 vs 0.029 random**, and falling with training |
| Superseded | the two 4,966-structure arms, kept further down as history |

Run history, so the numbers below are attributable:

| run | data | epochs | centering | best | where |
|---|---|---|---|---|---|
| baseline | 4,966 | 50 | off | epoch 17, val 562.167 | archived |
| centered | 4,966 | 50 | on | epoch 9, val 1071.370 | archived |
| **full** | **150,169** | **40** | **on** | epoch 40, val 596.319 | `/workspace/checkpoints/best.pt` |
| short | 150,169 | 10 | on | epoch 10, val 594.084 | `/workspace/ck10/`, epochs 5 and 10 kept |
| Next run | full PDB — 154,500 structures, uncapped, `--position-centering`, 15 epochs |

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
distinct proteins.

> **This paragraph used to continue: "and it is a dataset-size problem, not a
> loss problem — scaling to the 21,598 clusters in `pdb_ids.txt` is what
> addresses it." That prediction has been tested and is WRONG.** At 21,265
> clusters the gap is **48.6×**, marginally worse. See *The full run* below. The
> claim is recorded rather than deleted because it is what justified spending a
> night of GPU time.

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
reads 0.009 untrained, 0.008 for an identity-only encoder and 1.000 for a
position-only one.

> **This section used to say "where they disagree, trust `free`, because a
> permuted chain is not a protein and `shuf` asks the encoder about an input
> unlike anything it trained on." That guidance is now UNSUPPORTED.** On the full
> run they disagree by 7× — `free` 0.058, `shuf` 0.406 — and the independent
> evidence (TEST 5) sides with `shuf`. The reasoning above is still true as far as
> it goes; it simply is not a reason to disbelieve `shuf` when `shuf` is
> corroborated. See *What the two shortcut metrics actually caught* below.

At the time the two 4,966-structure arms were written up, **`shuf` had never been
read on a trained checkpoint** — TEST 7 landed after that pod was provisioned, so
`--test shortcut` hit a usage error on its older clone and the pod was terminated.
Every `shuf` number in this section is from a synthetic or untrained encoder, and
both arms' conclusions rest entirely on `free` and TEST 5. That gap is why
centering was declared a success on evidence that could not have detected its
main limitation.

---

## The centering arm: it prevents the shortcut `free` can see, and costs nothing

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
1,547 distinct proteins against 43M parameters — and it is what scaling to 21,598
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

**Decision: run the full build with `--position-centering`.** It prevents the
shortcut at no measured cost, and the open question — whether the shortcut
re-emerges at 14× the diversity — is the one thing this dataset cannot answer.

**Outcome, since resolved:** it was run that way, and `free` did stay contained
(0.058 at epoch 40). But `shuf` read 0.406 on the same checkpoint, so the shortcut
was contained only in the linear component `free` can see. The decision was right
on the evidence available and the evidence available was too narrow.

---

## The scale-up: what is actually in `pdb_ids.txt`

Measured on the committed `pdb_ids.txt` and `pdb_clusters.txt`:

| | |
|---|---|
| structure IDs | **154,500** (all with a cluster assignment) |
| distinct proteins at 30% identity | **21,598** |
| redundancy | **7.2×** |
| clusters contributing exactly one structure | **8,023** |
| share of all structures from the top 1,000 clusters | **48.5%** |
| disk, built (`.npz`, streamed — no `.cif` kept) | **0.43 GB** |

The redundancy is why `--max-per-cluster` exists, and the table below is the
reason it is **not** being used on the next run:

| cap | structures | clusters | steps/epoch |
|---|---|---|---|
| 1 | 21,598 | 21,598 | ~1,230 |
| 5 | 59,198 | 21,598 | ~3,373 |
| 10 | 79,406 | 21,598 | ~4,525 |
| 20 | 98,831 | 21,598 | ~5,632 |
| none | 154,500 | 21,598 | ~8,800 |

**Every cap yields the same 21,598 distinct proteins.** Capping buys no diversity
at all; it only drops near-duplicate deposits — point mutants, ligand complexes,
alternative conformations, better resolutions.

Three things follow, and the third is the one that decided it:

- **Disk is irrelevant.** 0.43 GB at the largest setting. An earlier version of
  POD_SETUP capped partly on storage grounds; that reasoning was simply wrong.
- **Compute is close to irrelevant too, if compared at fixed gradient steps rather
  than fixed epochs.** At ~132,000 steps, uncapped gives 15 passes over 154,500
  structures and `--max-per-cluster 5` gives 40 passes over 59,198 — the same GPU
  hours either way.
- **The duplicates are mild augmentation, not noise.** Different crystal forms of
  the same protein are genuinely different contact maps. More distinct inputs seen
  fewer times each is the better shape against memorisation, and memorisation is
  this model's measured failure.

The honest cost of going uncapped is imbalance: half the training time goes on 5%
of the families. That is a real objection and it is unmeasured — nothing here says
which effect wins. It is recorded so that a disappointing next run has a named
suspect.

---

## The full run: 150,169 structures, 40 epochs, centering on

2026-08-05. 135,147 train / 15,022 val over 21,265 clusters, 9,756 steps/epoch,
390,240 steps, ~6 h on a 4090. `--position-centering --amp-dtype bf16 --seed 0`.

### 1. More data did NOT close the train/val gap

This was the entire premise of the scale-up, and it failed:

| | 4,966-structure run | **150,169-structure run** |
|---|---|---|
| distinct proteins | 1,547 | **21,265** |
| train / val `on_diag` gap | 42× | **48.6×** |

**14× the diversity, and the gap is slightly worse.** Whatever this is, it is not
a dataset-size problem. That was the leading hypothesis in this file for two
weeks and it is now retired. The remaining candidates — untested — are model
capacity against an objective that does not constrain enough, and the objective
itself admitting solutions that do not transfer.

### 2. Checkpoint selection is on the wrong quantity, and it cost this run

Total val loss fell monotonically, 680 → 596, so selection kept promoting later
epochs. But total loss is `on_diag + 0.005 × off_diag`, and the two halves moved
in opposite directions:

| | epoch 1 | **epoch 8** | epoch 40 |
|---|---|---|---|
| val `on_diag` (agreement) | 465 | **400** ← best | 453 |
| val `off_diag` (decorrelation) | 43,035 | 42,191 | **28,577** |
| val total | 680 | 611 | **596** ← selected |

**Every bit of val improvement after epoch 8 came from the off-diagonal term.**
Agreement — the thing the objective exists to produce — got 13% worse over the
following 32 epochs. `best.pt` is the checkpoint with the worst late-run
agreement.

This file has said since the first run: *"Judge on `on_diag` and P@L/5, never
total loss — total falls forever via the off-diagonal term."* The code never
implemented it; [train.py:1099](train.py#L1099) still selects on total val loss.
Written down, then not done.

Two mitigations were tested and neither rescued the run — see section 4. Fixing
selection is still correct, it just is not the explanation.

### 3. The result that holds: retrieval on families never seen

`--test retrieval --n-prot 100`, pool drawn from the val split, so no protein
here has a 30%-identity homolog anywhere in training:

| checkpoint | top-1 | top-3 | median rank | pooled z_map cosine |
|---|---|---|---|---|
| **random init** | **6%** | 9% | 31/100 | 0.985 |
| epoch 5 | 38% | 61% | 2 | 0.881 |
| epoch 10 | 44% | 60% | 2 | 0.830 |
| **epoch 40** | **54%** | **74%** | **1** | 0.652 |

Three things, in order of importance:

- **54% against a 6% random baseline**, on unseen families, with 99 distractors.
  This is the first unambiguous evidence that training produces something that
  generalises.
- **It rises monotonically with training** — 38 → 44 → 54. The most-trained
  checkpoint is the best one, despite the 48.6× loss gap. So "overfitting" in the
  loss sense does not mean the representation stopped improving.
- The last column is the off-diagonal term doing visible work: different proteins'
  pooled `z_map` vectors start at cosine 0.985 (indistinguishable) and separate to
  0.652 as training proceeds.

### 4. The result that does not hold: linear contact probe

`--test probe`, TEST 5, all on the same 60 fit / 30 held-out protein pool:

| checkpoint | P@L/5 (long-range) | AUC |
|---|---|---|
| epoch 5 | **0.036** | 0.616 |
| epoch 10 | 0.029 | 0.639 |
| epoch 40 | 0.028 | 0.636 |
| random encoder | 0.029 | 0.542 |
| distance-only | 0.038 | 0.758 |

**P@L/5 falls with training — exactly opposite to retrieval.** And the best
trained value ties the trivial distance-only prior.

Two corrections to how this table should be read:

- **"Pretraining added NOTHING" is too strong**, though that is what the tool
  prints. Its verdict keys only on P@L/5. On AUC every trained checkpoint
  separates cleanly from random — 0.616–0.639 against 0.542, consistent across
  three independent checkpoints. There is real contact-relevant signal; it is too
  diffuse to sharpen the top-L/5 predictions.
- **0.036 vs 0.029 is about 1.5 standard errors** on 30 held-out proteins. The
  epoch-5 advantage is nominal, not significant. It is also the reason the
  "select on `on_diag`" fix cannot be credited with much: at 0.036 it would still
  lose to distance-only.

Also worth recording: TEST 5 draws its pool with `randperm` over the **whole**
dataset, so ~90% of its "held-out" proteins are ones the model trained on. Not
fixed, deliberately — the fix would break comparability with every earlier
number in this file — but it means the probe numbers are, if anything, optimistic.

### 5. What the two shortcut metrics actually caught

| epoch | 1 | 6 | 20 | 40 |
|---|---|---|---|---|
| `free` | 0.158 | 0.086 | 0.062 | **0.058** |
| `shuf` | 0.551 | 0.451 | 0.433 | **0.406** |

`free` looks clean — 0.058, essentially the 0.065 the centered 4,966 arm gave, and
far below the 0.250 an uncentered run peaks at. `shuf` says **40% of `z_seq`
survives permuting the amino acids within the chain**, against 0.009 untrained.

Both are measuring positional content. They disagree by 7×, and the explanation is
mechanical: centering provably zeroes the **linear per-index mean**, which is
exactly the quantity `free` decomposes. It places no constraint on nonlinear or
higher-moment positional structure, and `shuf` says that is where the model went.

**So centering did not fail — `free` stopped being able to see the thing it was
installed to prevent.** The 4,966-structure experiment could not have caught this,
because `shuf` was never read on those checkpoints.

Unknown: whether 0.406 is bad. There is still no trained-model baseline for `shuf`
from a model anyone considers healthy. It could be that 40% positional content is
normal and harmless for a per-residue representation.

### 6. Where this leaves the model

Consistent story across all seven tests: **the model learns a protein-specific,
transferable signature of a chain, and does not learn transferable pairwise
contact geometry.** Retrieval (54% vs 6%), CKA (0.943 vs 0.059) and the `z_map`
separation all improve with training; the pair probe does not.

Those are not in conflict. Retrieval and CKA ask whether two representations of
the *same* chain are arranged alike, which is exactly what Barlow Twins optimises.
P@L/5 asks whether *which residues touch which* is linearly decodable from pairs
of residue vectors — a question the objective never poses.

---

## On changing the metric after seeing the result

Partway through the above, the argument was made that the linear pair probe is the
wrong bar: a downstream contact transformer is meant to do the pairwise reasoning,
so the encoder only owes it good per-residue chemistry, and one linear layer over
`[v_i+v_j, v_i*v_j]` cannot be expected to extract structure from a 512-d
embedding space.

That argument is sound on its merits and was not invented to explain the result.
It also arrived **after** a bad result on the metric it demotes, which is the
classic way to fool yourself. Recorded here so a future reader can weigh it.

A residue-level chemistry probe (TEST 8: amino-acid identity as a sanity floor,
contact number as burial, with a **one-hot amino acid** row as the bar that makes
it non-trivial) was specified in full and then set aside — the objection being
that a single linear layer is the wrong instrument for any question asked of this
embedding space. Nothing was built. The specification is in the session log if it
is wanted later.

What replaced it: TEST 4 was pointed at the val split and given `--n-prot`, which
is what produced section 3 above. That test asks the matching question directly —
do corresponding sequence/map pairs agree and non-corresponding ones not — with no
linear readout anywhere.

---

## The diagnostics were flattering the model

Three separate ways, all found in one session, all now known:

**1. TEST 4 retrieved from the training set.** It sampled its protein pool with
`randperm` over the whole dataset, so ~90% of what it retrieved was memorised.
Every checkpoint read 100% and the test could not rank them. Pointed at the val
split, the same checkpoints spread from 38% to 54% and random init drops to 6%.

**2. TEST 4 calls an untrained model HEALTHY.** Its verdict rule is
`top1 > 3 × chance` ([eval.py:409](eval.py#L409)). Random init scores 6% against
1% chance and passes. This is the same effect as the untrained CKA finding further
down — the architecture alone groups proteins. **The trained number is
uninterpretable without the random-init row beside it**, and the test does not
print one. Not yet fixed.

**3. TEST 5's held-out proteins are mostly training proteins.** Same `randperm`
flaw, not fixed, noted in section 4.

The pattern is one thing: **a diagnostic that does not print its own untrained
baseline will eventually be read as evidence.** The archived findings already
listed "give every metric an untrained baseline" as a measurement trap. It was
listed, and then three tests shipped without one.

---

## Operational traps, from running this at scale

Cheap to hit, expensive in wall-clock:

- **`--keep-epoch-ckpts` does not save every epoch.** It changes whether the
  periodic save is *kept*, not how often it happens; cadence is
  `CKPT_EVERY_EPOCHS = 5` ([train.py:149](train.py#L149)). A 10-epoch run with the
  flag yields epochs 5 and 10 only. Pass `--ckpt-every 1` too.
- **Without that flag, `last.pt` is a rolling file.** The 40-epoch run's epoch-8
  weights — the ones with the best val agreement — were overwritten and are
  unrecoverable.
- **Python block-buffers stdout when it is a file.** `nohup python train.py > log`
  shows nothing for minutes; the ~1.5 KB banner sits in an 8 KB buffer. Use
  `python -u`, or a hard kill loses the tail of the log.
- **`cd X && nohup ... &` backgrounds the whole list**, so the `cd` happens in a
  subshell and the log lands relative to `X` while your prompt never moves. Put
  the `cd` inside `bash -c`, and redirect to an absolute path.
- **`--ckpt-dir` overrides the `CKPT_DIR` env var** ([train.py:182](train.py#L182)),
  but the "saved best.pt" line prints only the basename, so there is no way to tell
  from the log which directory was written.
- **Epoch time does not scale with steps/epoch.** Estimating 32 min/epoch at 8,800
  steps from a measured 1 min/epoch at 286 steps overshot by ~4×, because the
  per-epoch validation pass, `free` and the probe are fixed costs that were most of
  that minute. Measure the first epoch; do not extrapolate.

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

## Done since the last revision of this list

- **Trained it, twice, then at scale.** Two 50-epoch arms on 4,966 structures, then
  40 epochs and 10 epochs on 150,169.
- **Homology-aware split.** `random_split` put 40.3% of val chains' exact sequence
  twins in train. Splitting is now grouped by 30%-identity cluster (0.8%), and
  checkpoints record which they used under `split.grouped_by_cluster`.
- **More data.** The 5,000-row RCSB query cap is gone; `pdb_ids.txt` holds 154,500
  IDs and `pdb_clusters.txt` their cluster assignments.
- **`shuf` read on trained checkpoints** for the first time — 0.406, and it
  disagrees with `free` by 7×.
- **TEST 4 retrieves from the val split** and takes `--n-prot`. This is what turned
  a saturated 100% into the 6% / 38% / 44% / 54% spread that is now the run's
  headline result.

## Next, in order

1. **Give TEST 4 a random-init row.** It currently calls an untrained model
   HEALTHY, and every retrieval number in this file is only meaningful against the
   6% baseline that has to be obtained by a separate invocation. Same pattern TEST
   5 already uses. This is small, and until it exists the best result in the file
   rests on a comparison the tool does not make.
2. **Select checkpoints on val `on_diag`, not total val loss.**
   [train.py:1099](train.py#L1099). The advice has been in this file since the
   first run. Cheap, correct, and worth roughly nothing on its own — do it so the
   next question is not confounded by it.
3. **The scratch control.** A contact head plus the same model trained from random
   init; **the delta is the result**. Now the single highest-value open item,
   because retrieval at 54% vs 6% shows the encoder learns *something*
   transferable, and nothing in this file establishes that the something is worth
   more than training the downstream model from scratch. Sharpened by
   `distance-only` beating the trained encoder on AUC, 0.758 vs 0.636.
4. **A `shuf` baseline from a model believed healthy.** 0.406 is currently
   uninterpretable — the only reference points are 0.009 untrained and 1.000
   synthetic-positional. Without a third point there is no way to tell whether 40%
   positional content is a defect or normal.
5. **Decide what the encoder is FOR, and test that.** The pair probe and the
   retrieval test disagree because they ask different questions, and the project
   has not committed to which one it needs. If the downstream contact transformer
   is the consumer, the honest test is to train it on frozen `z_seq` and compare
   against training it on one-hot sequence — item 3 in a different guise, and the
   only version that settles the argument.
6. **Only if a later run disappoints:** re-run with `--max-per-cluster 5` to test
   whether the 48.5%-from-1,000-clusters imbalance matters. Same seed, same step
   budget.

---

## Measurement traps

All of the traps in the [archived findings](FINDINGS_2026-07-26_jepa.md#measurement-traps)
still apply — they are properties of the metrics, not of the architecture. In
particular: never rank CKA across different sample counts, never mean-pool before
comparing, never compare the two branches by cosine, and give every metric an
untrained baseline.
