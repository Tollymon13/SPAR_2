import sys
sys.path.insert(0, 'main')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from barnett_rsa import J0_long, J1, B, B_vals, Goal, STICKS

sticks_np = np.array(STICKS)

betas_to_plot = [0, 2, 5, 10] # betas used for plotting
b_indices = {bv: int(np.where(np.array(B_vals) == float(bv))[0][0]) for bv in betas_to_plot} # beta indicies

# delta function representing beta value (i.e. judge knows beta exactly under Barnett's eq. 7)
def delta(b_idx):
    """Point-mass prior over beta: judge knows beta exactly (Barnett's assumption)."""
    d = np.zeros(len(B)); d[b_idx] = 1.0
    return jnp.array(d)

LONG_J1 = {}
for bv in betas_to_plot:
    J1_table = np.array(J1(delta(b_indices[bv])))  # shape (N, 2); computes posterior for both
    LONG_J1[bv] = J1_table[:, int(Goal.LONG)]   # P(long | v, LONG speaker, beta=bv); LONG only

# Plot
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ['#4393c3', '#f4a261', '#e76f51', '#7b1d0e']

ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
ax.plot(sticks_np, J0_long, 'k-o', lw=1.8, ms=5, label='J0 (literal)')
for bv, col in zip(betas_to_plot[1:], colors[1:]):
    ax.plot(sticks_np, LONG_J1[bv], '-o', lw=1.8, ms=5, color=col,
            label=f'J1 beta={bv}')
ax.set_xlabel('stick value u')
ax.set_ylabel('P(long | u revealed)')
ax.set_title('Single speaker (LONG-biased)\nliteral J0 vs pragmatic J1')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0.05, 0.95)

plt.tight_layout()
plt.savefig('figure3.pdf', bbox_inches='tight')
plt.savefig('figure3.png', bbox_inches='tight', dpi=150)
print('Saved figure3.pdf / .png')
