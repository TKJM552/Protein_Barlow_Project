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

1. **Widen `SEED_DIM`** (`map_encoder.py:28`) — multiple distance bands, local
   clustering coefficient, normalised sequence position. Highest leverage,
   cheapest to try, targets the diagnosed bottleneck.
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

Three of the five original diagnostics gave misleading verdicts on this run. All
three were measurement bugs, not model bugs. Fix the instrument before trusting
any experiment run through it.
