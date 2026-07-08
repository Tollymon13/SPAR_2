# CLAUDE.md — RSA Model of AI Debate (in memo)

## Motivation

Training modern AI systems depends heavily on human judgements — filtering data, red-teaming,
expressing preferences on generated answers. Alignment is fundamentally a human problem, yet AI
Safety research has abstracted away the study of human judgement, including biases, over-reliance
and cognitive limits.

We study this directly using computational cognitive models. We focus on **AI Debate**: a form of
collaborative deliberation where two AI debaters argue opposing sides for a human judge. The safety
guarantee rests on how the judge reasons, so we examine it with a computational cognitive model of
how the judge updates beliefs.

**Research question:** Can the human judge in AI Debate be analysed with a cognitive model,
specifically the Rational Speech Act framework, and what does that reveal about how they reason
toward or away from the honest answer?

## Methodology

We build on **Barnett et al. 2022** (Bayesian Persuasion), who extend the RSA framework with a
persuasion term to model a judge interpreting a single strategic speaker. We extend this to the
two-advocate debate setting inside the **memo** probabilistic programming language (Chandra et al.
2025), designed for agents reasoning about other agents.

**Key insight — Barnett's single-speaker = consultancy:** Barnett's Figure 3 shows a judge
updating on one LONG-biased advocate. This is exactly consultancy (one-sided advice). The mirror
image (SHORT-biased) gives the inverted curve. Our base system (J1_two with LONG vs SHORT) is
the debate extension of this same setup.

**Weak evidence effect:** Our first target result. A sceptical judge seeing a moderately strong
stick paradoxically updates toward the opposite conclusion — the pragmatic inference being that
if the speaker showed only moderate evidence, the full picture must be worse. Reproducing this
is the first validation step before extending to full debate.

We sweep across model settings: scepticism (β), judge prior type, evidence strength, protocols
(debate vs consultancy). The goal is not to show debate beats consultancy (empirical evidence
already shows this). The goal is to characterise how a judge's reasoning influences whether debate
moves them toward the honest answer.

---

## Barnett's world (the foundation)

A "world" w consists of K sticks, each drawn uniformly from `STICKS = [0.1, ..., 0.9]`.
The truth is whether the mean stick value is ≥ 0.5 (`is_long`). Prior is uniform over all N^K
worlds (N=9, K=5 for experiments, K=2 for validation).

**World encoding:** `w = Σ_j v_j · N^j` where `v_j ∈ {0..8}` is the value index of stick j.

**J0 (literal judge):** `J0_long[v]` = fraction of all N^(K-1) completions where mean ≥ 0.5.

**β grid:** `B_vals = [0, 2, 5, 10, 20]`, indexed by `B = jnp.arange(5)`.
β must reach 10+ for sharp cherry-picking.

---

## Core agents (memo)

### S1 — Strategic speaker (Barnett Eq. 8)
Picks stick j ∝ exp(β · util), where util = lnJ0_scaled[b_idx, val_j] for LONG,
lnJ0_short_scaled[b_idx, val_j] for SHORT.

### J1 — Pragmatic judge, single reveal (Barnett Eq. 7)
Inverts S1: given revealed value V and speaker goal g, infers P(is_long).
Marginalises over worlds and β using judge's prior over β.

### J1_two — Joint two-advocate judge (our extension, the "base system")
Two advocates (a1, a2) with goals g1, g2. a2 constrained to j2 ≠ j1 (without replacement).
Judge simultaneously conditions on both reveals, knowing both selection strategies.

- **Debate:** g1=LONG, g2=SHORT
- **Consultancy:** g1=LONG, g2=LONG (or equivalently, single LONG speaker = Barnett's setup)

**Key property:** Bayesian joint conditioning is commutative — no order effects.

---

## Judge types

| Judge | prior_b | Assumption |
|---|---|---|
| Naive | δ_0 | β=0; ignores cherry-picking entirely |
| Calibrated | δ_{b_true} | Knows true β |
| Inferring | flat over B | Infers β from observations |

---

## Status summary

See `PLAN.md` for full detail. Core model complete and validated. Two figures complete and
committed. Next: multiple judge types (Step 4), then parameter recovery (Step 5).

---

## Implementation notes

- **B indexing:** `B = jnp.arange(5)` (indices 0..4), `B_vals = [0,2,5,10,20]`.
  Never use B_vals as the memo axis.
- **lnJ0_scaled:** Precomputed `jnp.outer(B_vals, lnJ0)` to avoid B_vals indexing inside memo.
- **K=5** for all experiments. K=2 for validation only.
- **J1 signature:** `J1[V: StkIdx, g: Goal](prior_b)` → shape (N, 2). Position marginalised out.
- **World encoding:** `W = jnp.arange(N**K)`. Decode stick j in world w: `(w // N**j) % N`.
  figure files use `stick_idx_arr = (W_np[:, None] // N_pow[None, :]) % N` for all worlds at once.
- **Symmetric debate averaging:** Both `mat` (judge belief surface) and S1 weighting must be
  symmetrized over both picking orderings (LONG-first and SHORT-first) independently.
  Using only LONG-first for S1 weighting while `mat` is symmetric is inconsistent.
- **LONG = ground truth:** In all figures, LONG is the honest debater arguing truth. SHORT is
  always the liar. Figures filter to long worlds only to ask: does the judge reach truth?
- **mat antisymmetry check:** `mat[i,j] + mat[j,i] ≈ 1` for all i,j confirms the model
  treats both debaters symmetrically.

---

## Validation oracles (all pass, K=2)

1. S1 β=0 uniformity: S1(β=0) = 1/K for all worlds and goals
2. J1(δ_0) = J0: naive judge reduces to literal judge
3. J1(δ_5) matches Barnett Python reference to < 1e-6
