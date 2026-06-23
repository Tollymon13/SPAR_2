import numpy as np
import jax.numpy as jnp
from memo import memo
from enum import IntEnum

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


