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

**Where a new session should start:** the pretraining phase is finished and written
up. What is not done is the experiment that says whether any of it was worth doing
— `contact_predictor.py` and `train_contact.py` are built, self-tested and **never
run**. See *The contact predictor and the two-arm experiment* below, then item 1 of
*Next, in order*. Everything between those two points is background.

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

Two mitigations were tested and neither rescued the run — see section 4.

**And then the obvious fix turned out not to be obvious.** `--select-on
{total,on_diag}` was added, and the default deliberately stayed `total`, against
this file's own standing advice, because by then there was evidence:

| | prefers | agrees with |
|---|---|---|
| retrieval on unseen families (54% at epoch 40) | **later** checkpoints | total val loss |
| linear contact probe (0.036 at epoch 5) | **earlier** checkpoints | val `on_diag` |

Selecting on `on_diag` would have optimised for the metric that is failing and
against the one that is working. The flag exists so the choice is deliberate and
stamped into the checkpoint; **the standing advice to early-stop on `on_diag` is
withdrawn** until something establishes which metric matters. That is what the
contact run is for.

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

## The contact predictor, and what building it measured

Built 2026-08-06 as [contact_predictor.py](contact_predictor.py) and
[train_contact.py](train_contact.py). Not yet run. Recorded now because several
of the design decisions were settled by measurement rather than preference, and
those measurements are results in their own right.

### The shape of a contact map, measured over 300 proteins

| | |
|---|---|
| contacts at `\|i-j\| < 12` | **72.3%** |
| non-contacts per contact | **45.1** (33.6 over the batches `train_contact` samples) |
| short-range positives per long-range one | **2.61** (2.77 likewise) |

Both numbers are load-bearing, and neither was guessable:

- **45.1** is why the loss needs `pos_weight`. Answering "no contact" to every
  pair scores ~97.8% accuracy with a loss curve that falls convincingly.
- **72.3%** is why it needs a *second*, separate weight. Consecutive CA atoms sit
  ~3.8 A apart against an 8 A threshold, so proximity in the chain nearly forces
  contact — those pairs are free, and under a flat loss **72% of the gradient on
  positives lands on the band a model can get from `|i-j|` alone.**

The first instinct — drop the short band from the loss — is wrong, and was
corrected before it shipped: the model then cannot *predict* the short band, and
a contact map with a hole down its diagonal is not a contact map. `long_weight`
upweights the long-range pairs instead, and like `pos_weight` it is measured off
the data rather than picked.

### Whole proteins fit; cropping was never necessary

Cropping is the standard answer to an L^2 model and it was the first design here.
It has a cost that is easy to miss: **pairs further apart than the crop are never
a training target**, so a 192-crop model has never once seen a contact between
residues 400 apart. That is a cap on what can be learned, not just on what fits.

Two changes removed the need. Estimated activation memory for one protein at
`PAIR_DIM=64`:

| L | plain | + checkpointing | + symmetric |
|---|---|---|---|
| 218 (median) | 1.1 GB | 0.4 GB | 0.3 GB |
| 512 | 6.0 GB | 2.0 GB | 1.6 GB |
| **990 (longest)** | **22.3 GB** | **7.5 GB** | **6.0 GB** |

- **Gradient checkpointing** recomputes each axial block during backward instead
  of storing it: ~89 pair-sized tensors down to ~30, for ~30% more compute.
- **Batching by pairs, not residues.** `train.py`'s residue budget is correct for
  the encoder and bounds nothing in a pair track. 4 proteins of 1024 and 20 of
  205 are both 4096 residues but **4.2M vs 0.84M pairs** — five times the memory
  from batches that look identical by the count that is actually being enforced.

So `CROP = 0` is the default and training sees the same whole proteins evaluation
does. **These are estimates derived from reading the code, not measurements** —
fusion, early frees and bf16 all move them, so the first real run prints
`torch.cuda.max_memory_allocated()` and `--pair-budget` should be retuned from
that, not from the table.

### Width is the expensive dimension

`PAIR_DIM` is paid once per cell and there are L^2 cells, so it is not comparable
to the sequence track's 512. Memory is linear in it; the linear layers are
quadratic (C -> 4C -> C):

| PAIR_DIM | memory at L=990 | linear FLOPs |
|---|---|---|
| 64 (current) | 7.5 GB | 1x |
| 128 | 15.1 GB | 4x |
| 256 | 30.1 GB | 16x |
| 512 | 60.2 GB | **64x** |

512 does not fit. 128 is the sensible ceiling and is what AlphaFold2 uses for its
own pair track.

### Symmetry buys ~30%, exactly rather than approximately

A contact map is symmetric, so half the pair tensor is redundant. Storing only a
triangle does not pay — GPUs are fast on dense rectangles and the gathers needed
to rebuild rows cost more than the arithmetic they save. Exploiting it in the
*operations* does: if `p` is symmetric then

    column_attention(p) == row_attention(p).T

so running both computes the same thing twice. One pass plus a transpose is
identical output for half the attention work and 15 stored tensors per block
instead of 21.

The cost is that the pair tensor can then never hold an asymmetric feature at any
depth. **Unmeasured.** The argument it does not matter: the target is symmetric.
The argument it might: AlphaFold's pair track is deliberately asymmetric because
triangular updates encode a directed constraint (i -> k -> j), a mechanism this
model does not have. Hence `--symmetric-pair` is a flag and an ablation, not a
default.

---

## The contact predictor and the two-arm experiment

**Status: built, self-tested, never trained.** Two files, both committed, neither
has produced a result. This section is the handover.

### Why it exists

Every diagnostic in this file measures a proxy. Retrieval says the encoder learned
something transferable (54% vs 6% random); the linear pair probe says it did not
(0.028 vs 0.029). Neither answers the question the project is actually asking,
which is whether the pretrained encoder is worth more than feeding raw amino acids
to a downstream contact model.

That question has exactly one honest test, and it is a controlled comparison:

| | arm A -- `--arm pretrained` | arm B -- `--arm scratch` |
|---|---|---|
| input | frozen `z_seq` from a Barlow Twins `best.pt` | `nn.Embedding(21, 512)`, trained |
| input parameters | **18.9M, frozen** | **10,752, trained** |
| context in the input | six encoder blocks' worth | **none** |
| predictor | identical, 7.17M | identical, 7.17M |
| split / seed / epochs | identical | identical |

Arm B is a lookup table: every leucine in every protein gets the same 512 numbers,
with no idea what is next to it. All context has to be built by the predictor's own
two RoPE blocks. **Arm A gets a 1,750x larger input model, pretrained on 135,000
proteins. If it cannot beat the lookup table, the pretraining bought nothing** --
and that is a clean result, not a failure of the test.

The encoder is FROZEN in arm A on purpose. Let it fine-tune and a win no longer
separates "the representation was good" from "43.1M more trainable parameters were
good", and arm B has no matching 43.1M to offer.

### Reading the result, in this order

1. **Does EITHER arm beat distance-only on long-range P@L/5?** If not, nothing
   below matters -- both learned the `|i-j|` prior and stopped. This is a live
   risk: distance-only already beats the pretrained encoder on AUC, 0.758 vs 0.636.
2. **Does A beat B?** That is the pretraining question.
3. Only then, the absolute numbers.

### Architecture, in one table

`contact_predictor.py`. (B, L, 512) + mask -> (B, L, L) logits.

| stage | shape | note |
|---|---|---|
| `in_proj` | (L, 512) | **no compression** -- see below |
| 2x `TransformerBlock` | (L, 512) | reused from `seq_encoder`, RoPE |
| `pair_init` | (L, L, 64) | `W_a h_i + W_b h_j + relpos(i-j)`; L^2 starts here |
| 8x `AxialBlock` | (L, L, 64) | row attention, column attention, FFN |
| `out` | (L, L) | symmetrised, `0.5*(x + x^T)` |

**`SEQ_DIM = 512`, equal to the input, is a fairness constraint not a capacity
choice.** `in_proj` runs per residue before any mixing. Arm B's embedding holds 21
distinct vectors, so any width above 21 is lossless for it; arm A's `z_seq` is
512-d and the Barlow Twins off-diagonal term actively decorrelates dimensions, so
it likely uses most of that width. Narrowing here penalises precisely the arm under
test. (An earlier version had 256, justified as saving pair memory -- wrong:
`SEQ_DIM` never reaches an L^2 tensor, since `pair_init` maps `seq_dim -> PAIR_DIM`.)

Axial attention rather than full attention over the grid: all-cells-to-all-cells is
O(L^4). Rows then columns is O(L^3), and information still reaches anywhere in two
hops -- `(i,j)` to `(i,k)` along a row, `(i,k)` to `(l,k)` down a column.

### The three ways this model looks healthier than it is

**1. Class imbalance.** Measured over 300 real proteins: **45.1 non-contacts per
contact**. Answering "no contact" everywhere scores ~97.8% with a healthy-looking
loss curve. `pos_weight_from_maps()` measures the ratio off the data.

**2. The `|i-j|` shortcut.** Also measured: **72.3% of all contacts sit at
`|i-j| < 12`**, because consecutive CA atoms are ~3.8 A apart against an 8 A
threshold. So under a flat loss nearly three quarters of the gradient on positives
lands on pairs obtainable from the offset alone.

The fix is NOT to drop the short band from the loss -- the model then cannot
predict it and the output has a hole down its diagonal. It is to rebalance:
`long_weight_from_maps()` returns the multiplier making both bands contribute equal
positive mass (**2.61** over those 300 proteins), and `separation_weight()` applies
it as a two-band step. **Train on all pairs, weighted; score on long range only.**

Run once with `--no-relpos` as the control that says how much of any score came
from the offset embedding rather than the sequence.

**3. Memory is L^2, not L.** A backward pass stores ~89 pair-sized tensors:

| | L=218 (median) | L=512 | L=990 (longest) |
|---|---|---|---|
| plain | 1.1 GB | 6.0 GB | **22.3 GB** |
| `grad_checkpoint=True` | 0.4 GB | 2.0 GB | **7.5 GB** |

Cropping is available (`--crop N`) but is NOT the default, because pairs further
apart than the crop are never a training target -- a cap on what the model can
learn, not just on memory. Instead: gradient checkpointing (~30% more compute),
plus `PairBudgetSampler`, which batches by **pair** count rather than residue count.
`train.py`'s residue budget is correct for the encoder and bounds nothing here --
4 proteins x 1024 and 20 x 205 are both 4,096 residues but 4.19M vs 0.84M pairs.

Depth in the pair track is nearly free under checkpointing (one saved boundary
tensor per block, not the ~21 computed inside), which is why it is 8 blocks. WIDTH
(`PAIR_DIM`) is the expensive dial.

### Running it

```bash
python train_contact.py --arm scratch    --seed 0 --epochs 20 --amp-dtype bf16
python train_contact.py --arm pretrained --seed 0 --epochs 20 --amp-dtype bf16 \
       --encoder-ckpt /workspace/checkpoints/best.pt
```

Same `--seed` and `--epochs` in both, or the comparison is void. `--smoke-test`
does one train step plus a few eval proteins and asserts the frozen encoder
receives no gradients.

`best.pt` here is selected on **val P@L/5**, not on loss -- deliberately not
repeating the selection that picked the worst-agreement checkpoint on the 150k run.

The banner prints measured `torch.cuda.max_memory_allocated()` after the first
epoch. **Retune `--pair-budget` from that number, not from the table above**, which
is derived from reading the code rather than from a GPU.

### Verified, and not

Verified on the committed 4,966-structure dataset: both arms smoke-test clean; the
frozen encoder receives no gradients; gradient checkpointing leaves gradients
bit-identical (check `j`); a single protein can be overfitted to loss 0.016 with a
planted long-range contact recovered (check `g`); measured `pos_weight` 33.4 and
`long_weight` 2.12 on that dataset.

Not verified: anything on a GPU, any real training run, any number that would go in
a results table. **No arm has been run.**

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

## The long-range threshold is not the field's

`eval.py` calls `|i-j| > 12` long-range and used to print "the field-standard
metric". It is not. The contact-prediction literature calls **12..23
medium-range** and reserves **long-range for `|i-j| >= 24`**.

Nothing internal is affected — trained, random and distance-only are all scored
identically — but **every P@L/5 in this file sits on an easier bar than a
published one** and none of them are quotable next to an external number. The
claim is removed from the code, and `train_contact.py` reports both cuts every
epoch so the stricter one exists from the first run.

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
- **TEST 4 prints its own random-init row** and judges against it rather than
  against chance, so it can no longer call an untrained model HEALTHY.
- **`--select-on {total,on_diag}`** exists, and the default deliberately stayed
  `total` against this file's own earlier advice — see below.
- **The contact predictor and its two-arm training script**, above.

## Next, in order

1. **Run both arms of `train_contact.py`.** Same `--seed`, same `--epochs`, one
   with `--arm pretrained --encoder-ckpt <150k best.pt>` and one with
   `--arm scratch`. **The delta is the result** — and it is the only thing that
   settles the argument this file has been circling for weeks. Retrieval at 54%
   vs 6% shows the encoder learns *something* transferable; nothing here shows it
   is worth more than feeding raw amino acids to the same predictor.

   Read the epoch lines in this order: (a) does **either** arm beat the
   `dist` column, (b) does pretrained beat scratch, (c) absolute numbers. Skipping
   (a) is how you conclude something from a model that learned `|i-j|`.

2. **`--no-relpos` once, on whichever arm wins.** The offset embedding hands the
   model the prior that already beats the pretrained encoder on AUC (0.758 vs
   0.636). If the score barely moves without it, the table was doing the work.

3. **A `shuf` baseline from a model believed healthy.** 0.406 is still
   uninterpretable — the only reference points are 0.009 untrained and 1.000
   synthetic-positional. Without a third point there is no way to tell whether 40%
   positional content is a defect or normal. The contact run supplies one for
   free: if the pretrained arm wins downstream *while* sitting at `shuf` 0.406,
   then 0.406 is survivable and the metric was over-read.

4. **`--symmetric-pair` as an ablation**, after the arms. ~30% cheaper for
   provably identical maths given a symmetric pair tensor; the only question is
   whether never holding an asymmetric intermediate costs anything.

5. **Only if the arms disappoint:** re-run the encoder with
   `--max-per-cluster 5` to test whether the 48.5%-from-1,000-clusters imbalance
   matters, and/or `--select-on on_diag` to test whether checkpoint selection
   does. Both are cheap, both are unlikely, and neither is worth doing before
   there is a downstream number to move.

---

## Measurement traps

All of the traps in the [archived findings](FINDINGS_2026-07-26_jepa.md#measurement-traps)
still apply — they are properties of the metrics, not of the architecture. In
particular: never rank CKA across different sample counts, never mean-pool before
comparing, never compare the two branches by cosine, and give every metric an
untrained baseline.
