import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum
import itertools

# World generation
STICKS = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
N = 9    # distinct stick values
K = 2    # sticks per world — K=2 for validation; change to 3 for experiments

W   = jnp.arange(N**K)   # world indices 0 .. N^K - 1
Idx = jnp.arange(K)      # stick positions 0 .. K-1


def stick_idx(w, j):
    """Value index (0..N-1) of stick j in world w."""
    return (w // int(N**j)) % N


def stick_val(w, j):
    """Actual stick value (0.1..0.9) of stick j in world w."""
    return STICKS[stick_idx(w, j)]


def is_long(w):
    """True iff the mean of all K stick values is >= 0.5."""
    total = sum(stick_val(w, j) for j in range(K))
    return total / K >= 0.5


def piw(w):
    """Uniform prior: 1 / N^K for every world."""
    return 1.0 / float(N**K)


# Literal Judge

def _compute_J0(K, sticks):
    sticks_np = np.array(sticks)
    N = len(sticks_np)

    # Possible N^(K-1) completions sampled with replacement.
    completions = np.array(list(itertools.product(range(N), repeat=K - 1)))
    assert completions.shape == (N**(K-1), K-1)

    # Total value of the K-1 hidden sticks for each completion.
    hidden_sums = sticks_np[completions].sum(axis=1)
    assert hidden_sums.shape == (N**(K-1),)

    J0 = np.zeros(N)
    for v in range(N):
        # Mean of revealed stick v and all hidden completions.
        means = (sticks_np[v] + hidden_sums) / K
        assert means.shape == (N**(K-1),)
        # Fraction of completions where mean >= 0.5. This is J0(long | v).
        J0[v] = np.mean(means >= 0.5)
    return J0


J0_long    = _compute_J0(K, np.array(STICKS)) # P(long | revealed=v)
lnJ0       = jnp.array(np.log(np.clip(J0_long,      1e-12, 1.0)))  # log P(long | v)
lnJ0_short = jnp.array(np.log(np.clip(1 - J0_long, 1e-12, 1.0)))  # log P(short | v)


