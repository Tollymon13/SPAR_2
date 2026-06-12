import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum

# --- Domain arrays ---
K   = 5                          # sticks per world
NB  = 3 ** K                     # worlds per truth value (243)
W   = jnp.arange(2 * NB)        # all worlds: 0..485
Idx = jnp.arange(K)             # stick indices 0..4
Len = jnp.array([1, 2, 3])      # stick lengths
B   = jnp.arange(0, 21)         # β grid: 0..20 (must reach tens; see CLAUDE.md)

# --- Evidence regimes ---
# PL is the likelihood model P(length | X)

# AMBIGUOUS: weak separation — the regime where the spiral lives.
# A one-sided debater can keep finding true-but-misleading sticks.
PL = np.array([[0.45, 0.30, 0.25],
               [0.25, 0.30, 0.45]])

# SEPARATED: strong separation — for evidence-regime comparisons only.
PL_SEPARATED = np.array([[0.70, 0.20, 0.10],
                          [0.10, 0.20, 0.70]])

# --- Truth convention ---
# X=1 is always the true world state. Behavioural data is relabelled to match.
X_TRUE = 1

# --- World encoding ---
# w = X * NB + base3(lengths), where (length - 1) is stored as a base-3 digit.

def Xof(w):
    """Truth value of world w (0 or 1)."""
    return w // NB

def stick(w, j):
    """Length of stick j in world w (1, 2, or 3)."""
    return ((w % NB) // (3 ** j)) % 3 + 1

def is_long(w):
    """True iff world w has X=1 (the true side by convention)."""
    return Xof(w) == 1

# --- Structured prior ---
# P(w) = P(X) * P(sticks | X) = 0.5 * prod_j PL[X, len_j - 1]
#
# This is the Bayesian prior: P(X) is uniform (0.5), P(sticks | X) is the
# likelihood of those stick lengths under truth X. 
_prior = np.array([
    0.5 * np.prod([PL[w // NB, ((w % NB) // (3 ** j)) % 3] for j in range(K)])
    for w in range(2 * NB)
])
prior_arr = jnp.array(_prior)

def piw(w):
    """Structured prior P(w) — callable inside @memo."""
    return prior_arr[w]

# --- Precomputed lookup tables (numpy, for use outside memo) ---
truth_arr = np.array([int(Xof(jnp.array(w))) for w in range(2 * NB)])
slen_arr  = np.array([[int(stick(jnp.array(w), j)) for j in range(K)]
                       for w in range(2 * NB)])

# --- Sanity checks ---
assert float(prior_arr.sum()) > 0.999, "Prior does not sum to 1"
assert int(stick(jnp.array(0), 0)) == 1, "stick(0,0) should be 1"

# ── Block 2: debater ────────────────────────────────────────────────────────
# Barnett speaker (Eq. 8) — independent debater used for a1, a3, and all
# consultancy debaters.

class Goal(IntEnum):
    SHORT = 0   # argue for X=0
    LONG  = 1   # argue for X=1 (truth by convention)

_PL_jax = jnp.array(PL)  # JAX version of PL for dynamic indexing inside @memo

def util(length, g):
    """Barnett Eq. 8: log P(X=g | length) under L0, computed from PL via Bayes.
    Numerator = PL[g, length-1]; denominator = total probability that any stick has length L, regardless of truth (X=0 or X=1), so marginalisation over X
    Result is in log-space so that
    exp(β · util) = prob^β — a soft-max over persuasiveness.
    """
    prob = _PL_jax[g, length - 1] / (_PL_jax[0, length - 1] + _PL_jax[1, length - 1])
    return jnp.log(jnp.clip(prob, 1e-12, 1.0))

@memo
def debater[w: W, b: B, g: Goal, j: Idx]():
    adv: knows(w, b, g)
    adv: chooses(jj in Idx, wpp=exp(b * util(stick(w, jj), g)))
    return Pr[adv.jj == j]
