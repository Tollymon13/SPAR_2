import sys
sys.path.insert(0, 'main')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from barnett_rsa import S1, J1, B, B_vals, Goal, STICKS, N, K, W

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

# Plot 
fig, ax = plt.subplots(figsize=(6, 5))

data = [consultancy_beliefs[bv][is_long_arr] for bv in betas] # keep only LONG worlds
x = np.arange(len(betas))

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
ax.set_title('Consultancy (single LONG speaker)\nJudge belief in long worlds')
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('figure4.pdf', bbox_inches='tight')
plt.savefig('figure4.png', bbox_inches='tight', dpi=150)
print('Saved figure4.pdf / .png')
