# Plan: RSA Model of AI Debate

## Research Goal

Show that the human judge in AI Debate can be analysed with the RSA framework using memo/JAX,
and characterise how their reasoning influences whether debate moves them toward the honest answer.

**Key framing:** Barnett's single-speaker setup IS consultancy (one biased advocate). The mirror
image with a SHORT-biased speaker gives the inverted curve. Our debate extension (J1_two with
LONG vs SHORT) is the natural two-advocate counterpart. All comparisons are debate vs consultancy.

---

## Implementation Status

### Foundation (barnett_rsa.py) — COMPLETE

| Block | Status | Notes |
|---|---|---|
| Block 1: world | **DONE** | STICKS 0.1..0.9, encoding, is_long, piw |
| Block 2: J0 precomputed | **DONE** | numpy table, verified |
| Block 3: S1 speaker | **DONE** | Barnett Eq. 8 in memo |
| Block 4: J1 single-reveal | **DONE** | Barnett Eq. 7 in memo |
| Block 5: validation | **DONE** | 3/3 checks pass (< 1e-6 vs reference) |
| Block 6: J1_two | **DONE** | Joint two-advocate judge (debate + consultancy) |

Current K=5 (59,049 worlds). K=2 used for validation only.

---

## Research Steps

### Step 1 — Replicate Barnett ✓ DONE
Faithful reimplementation of J0, S1, J1 in memo. Validated against Barnett's reference to < 1e-6.

### Step 2 — Base system: joint two-advocate judge ✓ DONE
J1_two: joint Bayesian update on two advocates with opposing goals (debate) or same goal
(consultancy). Without-replacement selection (j2 ≠ j1). No order effects.

### Step 3 — Replicate Barnett's simulations with the base system ✓ DONE

**judge_update_comparison.py** — belief update curves (renamed from figure3.py)
- Panel 1: Consultancy baseline — literal J0 vs pragmatic J1 at β=0,2,5,10 (single LONG advocate)
  Replicates Barnett's Figure 3 exactly. Shows weak evidence effect.
- Panel 2: Debate — literal vs pragmatic J1_two, grouped by mean reveal (V1+V2)/2
  Symmetric averaging over all K*(K-1) position pairs and both picking orderings.
  Shows weak evidence effect disappears; high β makes judge more decisive.

Key finding: debate makes judge more responsive to evidence (steeper S-curve), not less.
But responsiveness ≠ accuracy — that's what honesty_check.py addresses.

**honesty_check.py** — single-round belief distributions in long worlds (renamed from figure4.py)
- LONG is always the ground truth; SHORT is always the liar.
- For each of ~30,000 long worlds: computes expected judge belief weighted by S1 probabilities.
- Box plots show distribution of judge beliefs across long worlds at β=2,5,10.
- Panel 1: Consultancy (single LONG speaker) — judge tends toward truth but with weak evidence
  effect creating outliers at high β (some long worlds get low beliefs).
- Panel 2: Debate (LONG vs SHORT) — judge more consistently above 0.5; high β helps.
- Both `mat` and S1 weighting symmetrized over both picking orderings independently.
- Currently calibrated judge only — needs judge type extension (Step 4).

### Step 4 — Multiple judge types TODO
For fixed true β, compare naive / calibrated / inferring judges across debate and consultancy.

| Judge | prior_b | Question |
|---|---|---|
| Naive | δ_0 | Does debate help a judge who ignores cherry-picking? |
| Calibrated | δ_{b_true} | Best case — knows true β |
| Inferring | flat over B | Realistic — infers β from observations |

Add to figure4: separate box plots per judge type, both protocols.

### Step 5 — Parameter recovery TODO
Generate synthetic judges with known parameters. Simulate their belief updates. Fit to recover
parameters using Barnett's fitting methods.

Parameters to recover: β (scepticism), recursion depth (L0/L1/L2), α (affiliation bias).
Show which parameters are identifiable and which are not.
Distinguish judge types: over-sceptical, well-calibrated, under-sceptical, biased.

### Step 6 — Strong argument experiment TODO
How does an L2 speaker (who models the L1 judge) change the debate dynamics?

- Single-debater: L2 LONG speaker vs L1 judge — does the judge get exploited?
- Two-debater: L2 LONG vs L2 SHORT — does the adversarial structure defend against exploitation?

### Step 7 — Split scepticism / affiliation experiment TODO
Judge has a different β per debater — trusting one advocate more than the other.
Implement separate β_1, β_2 in J1_two. Show how affiliation bias distorts the posterior.

### Step 8 — Multi-round dynamics TODO (gated by Steps 3–4)

Two update rules, implemented and compared:

**8a — Ideal Bayesian (joint J1_two):**
At each round, judge conditions jointly on all reveals so far against the original uniform world
prior. No scalar collapse — full posterior over worlds maintained. No order effects.
This is the theoretical ceiling: even a perfect judge, does debate help?

**8b — Scalar collapse (Barnett-style sequential):**
After each round, judge collapses their belief to a scalar P(long). Next round uses that scalar
as the new prior and applies J1/J1_two again. Loses world-level information between rounds.
Introduces order effects. More cognitively plausible — models how real judges reason.

Both track confident-wrongness cw(t) over T rounds.

Expected findings:
- Ideal Bayesian + debate → no spiral under any protocol (structural guarantee)
- Scalar collapse + consultancy → spiral (replicates Chandra et al.)
- Scalar collapse + debate → key open question: does debate's structure compensate for the
  information loss of scalar collapse, or does it still spiral?

**The gap between 8a and 8b is the main finding:** it quantifies what cognitive limitations
(information loss, order effects) cost in terms of accuracy, and whether debate is robust to them.

---

## Key Files

| File | Purpose | Status |
|---|---|---|
| `barnett_rsa.py` | Core model: J0, S1, J1, J1_two | Complete |
| `validate_rsa_vs_barnett.py` | Validation oracle (K=2, β=5) | Complete |
| `judge_update_comparison.py` | Belief update curves — consultancy vs debate (Step 3) | Complete |
| `honesty_check.py` | Judge belief in long worlds — does truth win? (Steps 3–4) | Complete (calibrated judge only) |

---

## Key Design Decisions

- **Consultancy = single speaker:** Barnett's J1 is the consultancy baseline. J1_two(LONG,LONG)
  is two-advocate consultancy. Both are one-sided. Debate = J1_two(LONG, SHORT).
- **K=5** for all experiments (59,049 worlds). K=2 for validation only.
- **Without-replacement** in debate: a2 constrained to j2 ≠ j1.
- **B = jnp.arange(5)**, B_vals = [0,2,5,10,20]. Indices, not values, as memo axis.
- **Judge prior:** δ_0 = naive, δ_{b_true} = calibrated, uniform = inferring.
