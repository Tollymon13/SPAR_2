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


_N_pow = jnp.array([N**j for j in range(K)])  # [1, 9, 81, ...] — concrete at trace time

def stick_idx(w, j):
    """Value index (0..N-1) of stick j in world w."""
    return (w // _N_pow[j]) % N


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



class Goal(IntEnum):
    SHORT = 0
    LONG  = 1

# beta grid — B is an index axis (0..4); B_vals holds the actual beta values
B_vals = jnp.array([0.0, 2.0, 5.0, 10.0, 20.0])
B = jnp.arange(len(B_vals))  # 0..4

# Precomputed scaled log-utilities: lnJ0_scaled[b_idx, val_idx] = B_vals[b_idx] * lnJ0[val_idx]
lnJ0_scaled       = jnp.outer(B_vals, lnJ0)
lnJ0_short_scaled = jnp.outer(B_vals, lnJ0_short)


def util(b_idx, val_idx, g):
    """Scaled log J0 utility for beta index b_idx, stick value index val_idx, goal g."""
    return jnp.where(
        g == int(Goal.LONG),
        lnJ0_scaled[b_idx, val_idx],
        lnJ0_short_scaled[b_idx, val_idx],
    )

# Strategic Speaker
@memo
def S1[w: W, b: B, g: Goal, j: Idx]():
    spk: knows(w, b, g)
    spk: chooses(jj in Idx, wpp=exp(util(b, stick_idx(w, jj), g)))
    return Pr[spk.jj == j]


StkIdx = jnp.arange(N)  # domain for conditioning on the revealed stick's value index

# Pragmatic Judge — single reveal

@memo
def J1[V: StkIdx, g: Goal](prior_b: ...):
    judge: knows(g, V)                                           # judge knows goal and revealed value
    judge: thinks[
        gen: chooses(w in W, wpp=piw(w)),                        # prior over worlds
        gen: chooses(b in B, wpp=array_index(prior_b, b)),       # judge's prior over beta
        spk: knows(gen.w, gen.b, g),
        spk: chooses(jj in Idx, wpp=S1[gen.w, gen.b, g, jj]())  # speaker picks a position
    ]
    # Judge sees the VALUE of whichever stick the speaker chose
    judge: observes_event(wpp=(stick_idx(gen.w, spk.jj) == V))
    return E[judge[Pr[is_long(gen.w)]]]


# Pragmatic Judge debate
# Protocol: debate = (g1=LONG, g2=SHORT)

@memo
def J1_two[J1: Idx, J2: Idx, V1: StkIdx, V2: StkIdx, g1: Goal, g2: Goal](prior_b: ...):
    judge: knows(g1, g2, J1, J2, V1, V2)
    judge: thinks[
        gen: chooses(w in W, wpp=piw(w)),
        gen: chooses(b in B, wpp=array_index(prior_b, b)),
        a1: knows(gen.w, gen.b, g1),
        a1: chooses(j1 in Idx, wpp=S1[gen.w, gen.b, g1, j1]()),
        a2: knows(gen.w, gen.b, a1.j1, g2),
        a2: chooses(j2 in Idx, wpp=(j2 != a1.j1) * S1[gen.w, gen.b, g2, j2]())
    ]
    judge: observes [a1.j1] is J1
    judge: observes [a2.j2] is J2
    judge: observes_event(wpp=(stick_idx(gen.w, J1) == V1) * (stick_idx(gen.w, J2) == V2))
    return E[judge[Pr[is_long(gen.w)]]]


