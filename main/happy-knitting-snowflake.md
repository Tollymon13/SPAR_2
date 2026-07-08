# Plan: Parameter Recovery — One Condition at a Time

## Goal

Implement synthetic parameter recovery in `main/parameter_recovery.py`.
Build up one condition at a time: β first, then judge type, then affiliation bias.
Each step is a separate commit with validation prints before moving on.

Recursive depth (L2) deferred — separate planning step after these three commits.

## Scientific argument

1. `judge_update_comparison` — theoretical: characterises the model (J1 curves per β)
2. `honesty_check` — theoretical: model predicts debate outperforms consultancy
3. `parameter_recovery` — empirical validation: the model is identifiable from behavioural data
4. → recovered parameters plugged back into (1) and (2) to profile a specific judge:
   their cognitive signature and their predicted accuracy in debate

This makes the case for applying the RSA model to real human judge data from AI debate.

---

## Noise model

Without noise, synthetic beliefs are deterministic (J1(V) is a fixed number) and recovery
is trivial. We add logistic-normal noise to simulate real response variability:

```
belief_observed = sigmoid(logit(J1(V)) + ε),  ε ~ N(0, σ²)
```

σ is fixed at 0.2 for synthetic validation. When applying to real data, σ would be
fit alongside β. Fitting uses MLE under this noise model (equivalent to Barnett's approach).

Log-likelihood of one observation:
```python
# residual on logit scale
logit_pred = np.log(J1_pred / (1 - J1_pred + 1e-10) + 1e-10)
logit_obs  = np.log(belief_obs / (1 - belief_obs + 1e-10) + 1e-10)
ll = -0.5 * ((logit_obs - logit_pred) / sigma)**2  # Gaussian on logit scale
```

---

## Commit 1 — β recovery: both consultancy and debate, calibrated judge

**File:** `main/parameter_recovery.py`

**Synthetic data generation (run for both protocols):**

*Consultancy:* single LONG speaker  
- For each β_true in B_vals = [0, 2, 5, 10, 20]:
  - For N=200 worlds (sampled with replacement from all N^K worlds):
    - S1(β_true) picks position j → reveal V = stick_idx_arr[world, j]
    - True belief = J1(delta(β_true))[V, Goal.LONG]
    - Noisy belief = sigmoid(logit(true_belief) + N(0, σ=0.2))
    - Store (V, noisy_belief)

*Debate:* LONG vs SHORT, symmetric over picking orderings  
- For each β_true in B_vals:
  - For N=200 worlds:
    - S1(β_true) picks j1 (LONG); S1(β_true) picks j2 ≠ j1 (SHORT, renormalised)
    - True belief = mat[v_{j1}, v_{j2}] from J1_two(delta(β_true))
    - Noisy belief = sigmoid(logit(true_belief) + N(0, σ=0.2))
    - Store (v_{j1}, v_{j2}, noisy_belief)

**Fitting:**
- For each β_true, fit β̂ by grid search over B_vals using MLE under the noise model
- Consultancy: predicted belief = J1(delta(β_cand))[V, Goal.LONG]
- Debate: predicted belief = mat[v_{j1}, v_{j2}] from J1_two(delta(β_cand))
- N_runs=20 independent noise realisations per β_true

**Validation print (both protocols):**
```
[Consultancy]
beta_true=0  -> beta_recovered=0   (20/20 runs correct)
...
[Debate]
beta_true=0  -> beta_recovered=0   (20/20 runs correct)
...
```

**Plot:** Two recovery matrix heatmaps side by side (consultancy | debate) — rows=β_true,
cols=β_recovered, values=fraction correct. Diagonal should be near 1 in both.
Saves as `recovery_beta.png`. Comparison shows whether debate makes β harder to identify.

---

## Commit 2 — Add judge type: naive vs calibrated vs inferring, both protocols

**Extended setup (run for both consultancy and debate):**
- Speaker β_speaker varies across session blocks: [2, 5, 10] (known to us, not to judge)
- For each (β_speaker_true, judge_type, protocol) combination:
  - Generate N=200 sessions with S1(β_speaker_true) picking reveals
  - Compute true judge belief using judge_type's prior_b:
    - Naive:      prior_b = delta(0)
    - Calibrated: prior_b = delta(β_speaker_true)
    - Inferring:  prior_b = uniform over B
  - Consultancy: J1(prior_b)[V, Goal.LONG]; Debate: mat[v_{j1}, v_{j2}] from J1_two(prior_b)
  - Add logistic-normal noise (σ=0.2)

**Fitting:**
- Joint grid search over (β_cand, judge_type_cand) — 5 × 3 = 15 candidates per protocol
- MLE: find candidate with highest log-likelihood given all sessions in a block
- N_runs=20 independent noise realisations

**Key identifiability:** β_speaker varies across blocks. Naive judge's beliefs don't change
between blocks; calibrated and inferring judges' beliefs do. This is what separates them.
In debate, two reveals per round give more information → expect faster/cleaner separation.

**Validation print (both protocols):**
```
[Consultancy]
judge_type=naive      beta_speaker=2  -> recovered=(beta=0,  naive)   (20/20)
judge_type=calibrated beta_speaker=2  -> recovered=(beta=2,  calib)   (20/20)
...
[Debate]
judge_type=naive      beta_speaker=2  -> recovered=(beta=0,  naive)   (20/20)
judge_type=calibrated beta_speaker=2  -> recovered=(beta=2,  calib)   (20/20)
...
```

**Plot:** Two sets of confusion matrices (consultancy | debate), one per judge type (3 panels
each). Saves as `recovery_judge_type.png`. Comparison shows whether debate's two-reveal
structure makes judge type easier or harder to distinguish.

---

## Commit 3 — Add affiliation bias: asymmetric β per debater

**New parameter:** (β_1, β_2) — judge's assumed β for LONG speaker and SHORT speaker.
Symmetric case: β_1 = β_2. Affiliation bias: β_1 ≠ β_2.

**Setup:**
- Debate sessions (LONG vs SHORT) using J1_two
- Test cases: symmetric (5,5), LONG-biased (5,2), SHORT-biased (2,5)
- Speaker reveals via S1(β_speaker) for both debaters
- True judge belief via J1_two with asymmetric prior_b per debater

**Requires barnett_rsa.py extension:**
J1_two currently takes a single prior_b shared across both debaters.
Extend signature to accept (prior_b_1, prior_b_2) — one per debater.
Localised change: split the beta marginalisation inside J1_two.

**Fitting:**
- Grid over (β̂_1, β̂_2) — 5×5=25 candidates
- Also fit constrained model (β_1=β_2) — 5 candidates
- Compare MLE of unconstrained vs constrained: is the asymmetry detectable?

**Validation:**
- Symmetric (5,5): unconstrained recovers (5,5); constrained fit not worse
- Asymmetric (5,2): unconstrained recovers (5,2); constrained fit has lower log-likelihood

**Plot:** Heatmap rows=β_1_true, cols=β_2_true, colour=log-likelihood gap between
unconstrained and constrained fit. Large gap = affiliation bias is detectable.
Saves as `recovery_affiliation.png`.

---

## Commit 4 — Downstream: plug recovered parameters into existing figures

**No new computation.** Refactor `judge_update_comparison.py` and `honesty_check.py`
to accept (β, prior_b) as arguments rather than hardcoded values.

Then in `parameter_recovery.py`, after fitting:
- Call `judge_update_comparison` with recovered (β̂, prior_b_hat) → shows this judge's
  belief curve: their cognitive signature (weak evidence effect strength, naivety)
- Call `honesty_check` with recovered (β̂, prior_b_hat) → shows predicted accuracy in
  debate vs consultancy for this specific judge profile

**Argument this enables:** "We fitted a judge's cognitive parameters from their responses
to reveals. Here is their belief curve. Here is what our model predicts they would do in
AI debate." This makes the case for applying the RSA model to real human judge data.

**Plot:** 2-panel summary for a fitted judge profile. Saves as `judge_profile.png`.

---

## File changes

```
main/
  parameter_recovery.py        # new file: Commits 1–4
  barnett_rsa.py               # Commit 3 only: extend J1_two for (prior_b_1, prior_b_2)
  judge_update_comparison.py   # Commit 4: accept (β, prior_b) as arguments
  honesty_check.py             # Commit 4: accept (β, prior_b) as arguments
```

---

## Commit structure

| # | Message |
|---|---|
| 1 | Add parameter_recovery.py: β recovery in consultancy and debate |
| 2 | Add judge type recovery: naive vs calibrated vs inferring, both protocols |
| 3 | Add affiliation bias recovery: asymmetric β per debater |
| 4 | Plug recovered parameters into judge_update_comparison and honesty_check |

Recursive depth (L2 speaker + J2 judge) deferred as separate plan.
