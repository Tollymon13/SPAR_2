import sys
sys.path.insert(0, 'main')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from barnett_rsa import S1, J1, J1_two, B, B_vals, Goal, STICKS, N, K, W

sticks_np = np.array(STICKS)
W_np = np.array(W)

# World setup (extract sticks from each of the N**K worlds)
N_pow = np.array([N**j for j in range(K)]) # base-9 as we have 9 possible values per stick 
stick_idx_arr = (W_np[:, None] // N_pow[None, :]) % N   # (N^K, K): value index per world per position (e.g. 11000 corresponds to (0.2, 0.2, 0.1, 0.1, 0.1)
vals_arr = sticks_np[stick_idx_arr] # (N^K, K): actual stick values
is_long_arr = vals_arr.mean(axis=1) >= 0.5 # (N^K,): ground truth per world (i.e. checks to see if it is a LONG world)

#  Precompute S1
print("Computing S1...")
S1_table = np.array(S1())   # (N^K, |B|, 2, K): P(pick position j | world, beta, goal)
print("Done.")

betas = [2, 5, 10]
b_indices = {bv: int(np.where(np.array(B_vals) == float(bv))[0][0]) for bv in betas}

def delta(b_idx):
    d = np.zeros(len(B)); d[b_idx] = 1.0
    return jnp.array(d)

# Consultancy
# For each world w (LONG or SHORT), compute judge belief assuming a LONG arguing speaker playing strategically 
consultancy_beliefs = {}
for bv in betas:
    b_idx = b_indices[bv]
    J1_table = np.array(J1(delta(b_idx))) # (N, 2): J1 per stick value per goal (i.e. Barnett's eq. 7: judge's belief after seeing a stick V)
    J1_long = J1_table[:, int(Goal.LONG)] # (N,): P(long | value, LONG speaker) (i.e. slice only the LONG section of judge's belief)

    s1_long = S1_table[:, b_idx, int(Goal.LONG), :] # (N^K, K): P(pick j | w, LONG) (i.e. probability that a LONG speaker picks position j)
    belief_per_pos = J1_long[stick_idx_arr] # (N^K, K): J1 at each position's value (i.e. judge's posterior after seeing the stick)
    consultancy_beliefs[bv] = (s1_long * belief_per_pos).sum(axis=1)  # (N^K,) (i.e. belief of the judge after seeting the stick multiplied by the probability of the LONG biased speaker picking that stick, summed over all K positions)
    # essentially it is the expected judge belief (i.e. in a world w, after one reveal from a LONG speaker, what is the judge expected to believe)

# Debate 
# For each world w: expected judge belief under debate (LONG vs SHORT) symmetric over all position pairs and both picking orderings
debate_beliefs = {}
for bv in betas:
    b_idx = b_indices[bv]
    Jt    = np.array(J1_two(delta(b_idx)))   # (K, K, N, N, 2, 2)

    # Symmetric debate belief surface: mat[v_long, v_short] = avg judge P(long) over all valid pairs
    long_first_mat  = Jt[:, :, :, :, int(Goal.LONG),  int(Goal.SHORT)]
    short_first_mat = Jt[:, :, :, :, int(Goal.SHORT), int(Goal.LONG)].transpose(0, 1, 3, 2)
    offdiag         = (1 - np.eye(K))[:, :, None, None]
    mat = ((long_first_mat + short_first_mat) * offdiag).sum(axis=(0, 1)) / (K * (K - 1) * 2)  # (N, N)

    s1_long  = S1_table[:, b_idx, int(Goal.LONG),  :]   # (N^K, K)
    s1_short = S1_table[:, b_idx, int(Goal.SHORT), :]   # (N^K, K)

    # Reveal indices for all (world, j1, j2) combinations
    V1 = stick_idx_arr[:, :, None]   # (N^K, K, 1): j1's stick value index
    V2 = stick_idx_arr[:, None, :]   # (N^K, 1, K): j2's stick value index
    offdiag_2d = (1 - np.eye(K))[None, :, :]   # (1, K, K): zero out j1==j2

    # LONG picks j1 first, SHORT renormalises over remaining positions
    excl_short   = s1_short[:, None, :] * (1 - np.eye(K))[None, :, :]  # (N^K, K, K): zero out j2==j1
    w2_short     = excl_short / (excl_short.sum(axis=2, keepdims=True) + 1e-30)  # (N^K, K, K): P(SHORT picks j2 | j1 taken)
    long_first   = (s1_long[:, :, None] * w2_short * mat[V1, V2] * offdiag_2d).sum(axis=(1, 2)) # expected judge belief in world w, where LONG picks first according to S1, SHORT picks second according to S1 (renormalised over the remaining positions), and the judge updates using posterior

    # SHORT picks j1 first, LONG renormalises over remaining positions
    # mat[V2, V1]: V2=SHORT's reveal → v_short axis, V1=LONG's reveal → v_long axis, so swap
    excl_long    = s1_long[:, None, :] * (1 - np.eye(K))[None, :, :]   # (N^K, K, K): zero out j2==j1
    w2_long      = excl_long / (excl_long.sum(axis=2, keepdims=True) + 1e-30)   # (N^K, K, K): P(LONG picks j2 | j1 taken)
    short_first  = (s1_short[:, :, None] * w2_long * mat[V2, V1] * offdiag_2d).sum(axis=(1, 2)) # expected judge belief in world w, where SHORT picks first according to S1, LONG picks second according to S1 (renormalised over the remaining positions), and the judge updates using posterior

    debate_beliefs[bv] = 0.5 * (long_first + short_first) # equal ordering is likely

# Plot 
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

for ax, beliefs, title in zip(
    [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])],
    [consultancy_beliefs, debate_beliefs],
    ['Consultancy (single LONG speaker)', 'Debate (LONG vs SHORT)']
):
    data = [beliefs[bv][is_long_arr] for bv in betas] # keep only LONG worlds
    x    = np.arange(len(betas))

    ax.boxplot(data, positions=x, widths=0.4,
               patch_artist=True,
               boxprops=dict(facecolor='#e76f51', alpha=0.7),
               medianprops=dict(color='darkred', lw=2),
               whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
               flierprops=dict(marker='o', ms=2, alpha=0.3))

    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'beta={bv}' for bv in betas])
    ax.set_ylabel('Judge belief P(long | reveal)')
    ax.set_title(f'{title}\nJudge belief in long worlds')
    ax.set_ylim(-0.05, 1.05)

plt.suptitle('Does truth win? Judge belief across long worlds (K=5)', y=1.02)
plt.tight_layout()
plt.savefig('figure4.pdf', bbox_inches='tight')
plt.savefig('figure4.png', bbox_inches='tight', dpi=150)
print('Saved figure4.pdf / .png')
