# ARCHIVED — Findings from the 50-epoch run, 26 July 2026 (JEPA architecture)

> **These numbers do not describe the current model.** They were produced by an
> architecture that no longer exists, which differed in two ways:
>
> 1. **A predictor head.** The sequence branch ran through a per-residue MLP whose
>    output was matched against the map branch. That head is gone; the loss now
>    compares the sequence encoder's own output (`z_seq`) with the map encoder's
>    (`z_map`) directly, and the objective is symmetric — nothing is a "target".
> 2. **A 2-scalar map seed** (local degree, long-range degree), diagnosed below as
>    the run's ceiling. Each residue is now seeded with its own contact-map row,
>    indexed by relative offset.
>
> Kept because the **measurement traps** section at the bottom is architecture-
> independent and every trap in it was paid for. Read that; treat every number
> above it as history. Live notebook: [FINDINGS.md](FINDINGS.md).

---

# Findings — 50-epoch run, 26 July 2026

First real training run. RTX 4090, ~1 min/epoch, 50 epochs in under an hour,
bf16, zero non-finite skips. Checkpoint `best.pt` (epoch 50, val_loss 73.212)
is not in git — logs are in `runs/2026-07-26_50ep_4090/`, also gitignored.

Written down because none of it is derivable from the code or git history.

---

## What the model learned

**Sequence→structure correspondence is real and protein-specific.**

| Measurement | Result | Baseline |
|---|---|---|
| Shuffled control (TEST 2) | real 69.0 vs shuffled 537.6 | — |
| CKA, pred vs target (TEST 6) | **0.923** | 0.042 random init, 15/15 proteins |
| CKA retrieval (TEST 4) | **68% top-1**, 88% top-3 | 4% chance |
| CKA, matched vs mismatched | 0.900 vs 0.312 | — |
| Linear probe P@L/5, long-range (TEST 5) | 0.128 | 0.041 distance-only, 0.025 random, 0.018 base rate |
| CKA on **unseen** RCSB structures | 0.82–0.92 | 0.03 untrained |

The last row matters most: it holds on structures deposited decades after the
training data (`test_novel.py`). That separation is far too large to be an artifact.

**Protein-level pooled vectors are unusable.** Mean-pooling collapses different
proteins' targets onto ~0.89–0.99 cosine to each other. Identity lives in the
residue *arrangement*; averaging destroys it. Downstream code must use
residue-level representations or a shape-aware readout (CKA), never `.mean(0)`.
Centering before pooling does not rescue it (tested: median rank 27→21 of 60).

**`pred` and `target` do not share a basis.** Per-residue cosine between them is
~0.00. This is expected, not a failure: Barlow Twins pushes each through a
*separate* expander and only constrains dimension-wise correlation of the
outputs. Measured post-expander, that correlation is **0.990, with all 2048
dimensions above 0.9** — the objective is nearly maximally satisfied.

---

## Why it stopped improving

Decomposing val loss over epochs 36–50:

```
val TOTAL   : -0.299/epoch     <- looks like steady progress
val on_diag : +0.0026/epoch    <- flat. the term that measures prediction.
val off_diag: -0.302/epoch     <- 100% of the apparent improvement
```

All late progress was the redundancy term decorrelating dimensions. The
invariance term plateaued around epoch 36, while the generalisation gap widened:

```
epoch 36: train on_diag 8.94 vs val  9.81   (gap +0.88)
epoch 50: train on_diag 7.55 vs val 10.01   (gap +2.47)   +0.073/epoch
```

**More epochs of this setup will not help.** The objective is ~93% solved
(cross-correlation diagonal 0.93, off-diagonal RMS 0.055) while P@L/5 is 0.128.
The ceiling is not optimisation or compute.

**Root cause hypothesis: `SEED_DIM = 2` in `map_encoder.py`.** The target branch
builds each residue from two scalars (local degree, long-range degree). Its
outputs come out homogeneous — mean pairwise cosine 0.42, effective rank 201 vs
pred's 245 — so matching them is easy and doesn't force the sequence branch to
encode much. Everything downstream inherits that ceiling.

Also relevant: 86% of the loss is `lambda*off_diag`, only 14% is prediction.

> **ACTED ON, 27 July 2026.** The 2-scalar seed is gone — each residue now enters
> the map encoder as its own contact-map row, indexed by relative offset `j − i`
> and projected `1999 → 512`. On one 581-residue chain the seed matrix goes from
> mean pairwise cosine 0.906 / rank 2 to 0.586 / rank 327. See the banner at the
> top of this file: this diagnosis was acted on, but the fix is unmeasured.

---

## What is NOT known

- **Contact-map prediction is untested.** The model has no `(L, L)` output.
  P@L/5 = 0.128 is what a *single linear layer* extracts from frozen embeddings —
  a lower bound on information content, not a prediction capability.
- **Whether JEPA pretraining beats supervised-from-scratch.** Nothing here
  compares against that. Beating random init is a low bar. This is the missing
  baseline, and no result in this file substitutes for it.
- **Homology leakage.** `random_split` puts homologous proteins in both train and
  val, so held-out numbers are optimistic. Any supervised work needs a
  sequence-identity or CATH-family split before its numbers mean anything.

---

## Next experiments, ranked

1. ~~**Widen `SEED_DIM`**~~ — **done, unmeasured.** Went further than "more
   summary statistics": the seed is now the raw contact row (see the note above).
   The run that tests whether this lifts the ceiling has not happened. Judge it on
   `on_diag` and on P@L/5, *not* on total loss or CKA — and run TEST 2 before
   believing any of it, for the reason in the last measurement trap below.
2. **Contact head + the scratch control.** A `(B, L, L)` output head with dilated
   2D convolutions, `BCEWithLogitsLoss(pos_weight≈50)` for the 1.8% base rate,
   symmetrised output. Reduce 512→~64 dims *before* going 2D — a
   `(B, L, L, 1024)` tensor at L=566, B=16 is 21 GB. Run it both from the JEPA
   checkpoint and from scratch; **the delta is the result**.
3. **More data.** `get_files.py:38` caps the RCSB query at `rows: 5000`; the PDB
   has ~200k. The widening gap says 4,966 is limiting.
4. **Early stopping on `on_diag`**, not total loss. Total falls forever via
   decorrelation and tells you nothing.
5. **`LAMBDA_OFFDIAG` 5e-3 → 1e-3** (`barlow_twins.py:26`). One line, rebalances
   gradient toward prediction. Ranked low: better agreement with an
   uninformative target buys little.
6. **Experiment tracking.** One log file per run does not survive item 1.

---

## Measurement traps (all hit at least once)

- **Never rank CKA values computed over different sample counts.** Fewer samples
  inflates CKA, so short decoys outrank the true structure. Fixed at `CMP_LEN`
  (150 residues) in `eval.py`. This bug inflated a reported 96% top-1 to its true
  68%.
- **Never mean-pool before comparing.** It reported 4% where CKA reports 68%.
- **Never compare `pred`/`target` by cosine.** No shared basis exists.
- **Chance MRR is `H_B/B`, not `1/B`** (0.211 vs 0.062 for B=16). The wrong
  baseline made an at-chance result look 3× chance.
- **TEST 1 must judge `on_diag`, not total loss.** With 4 proteins, N is far
  below `EXPANDER_DIM`, so `off_diag` has an unreachable floor and drags the
  total down with it. Reported "PARTIAL" where the truth was a 96% `on_diag` drop.
- **Every metric needs an untrained baseline.** CKA 0.87 looks unremarkable until
  you see random init scores 0.08.
- **A richer target can be a POSITION LEAK, and every agreement metric here would
  applaud it.** Found while building the row seed, before any training run. About
  4 of every 10 contacts sit at `|i−j| ≤ 2` and are present in 99.8% of those
  slots — consecutive CA atoms are ~3.8 Å apart, always inside the 8 Å threshold.
  Index the seed row by *absolute* partner index and that always-on band lands in
  columns `i−2..i+2`: a pure function of `i`, identical in every protein. Measured
  R² of predicting the target from position alone, 120 chains:

  | seeding | seed | encoder target |
  |---|---|---|
  | absolute index | 0.573 | 0.620 |
  | relative offset `j−i` | 0.031 | 0.033 |
  | old 2-scalar | 0.047 | — |

  62% of an absolute-indexed target is reproducible from RoPE position alone, so
  the sequence branch could satisfy Barlow Twins while learning no structure —
  and `on_diag`, CKA, and TEST 4 retrieval would all *improve*. **TEST 2 is the
  only diagnostic here that catches it**: a position-only solution survives
  rolling the maps, so the real-vs-shuffled gap collapses. Hence `SEED_MODE =
  "relative"`. Re-run check (g) of `python map_encoder.py` after any change to
  how the map is fed in.

Three of the five original diagnostics gave misleading verdicts on this run. All
three were measurement bugs, not model bugs. Fix the instrument before trusting
any experiment run through it — and note that the trap above is the first one
found in the *model* that the instruments would have rewarded rather than caught.
