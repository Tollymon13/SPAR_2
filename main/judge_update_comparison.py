import sys
sys.path.insert(0, 'main')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from barnett_rsa import J0_long, J1, J1_two, B, B_vals, Goal, STICKS, N, K

sticks_np = np.array(STICKS)

betas_to_plot = [0, 2, 5, 10]
b_indices = {bv: int(np.where(np.array(B_vals) == float(bv))[0][0]) for bv in betas_to_plot}

# delta function representing beta value (i.e. judge knows beta exactly under Barnett's eq. 7)
def delta(b_idx):
    d = np.zeros(len(B)); d[b_idx] = 1.0
    return jnp.array(d)

# Panel 1: Consultancy (single LONG speaker, Barnett Eq. 7)
LONG_J1 = {}
for bv in betas_to_plot:
    J1_table = np.array(J1(delta(b_indices[bv])))  # shape (N, 2): stick values x goals
    LONG_J1[bv] = J1_table[:, int(Goal.LONG)]      # P(long | v, LONG speaker, beta=bv)

# Panel 2: Debate (LONG vs SHORT, symmetric over all position pairs)
v1_idx, v2_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij') # all combinations of stick values
means = (sticks_np[v1_idx] + sticks_np[v2_idx]) / 2   # shape (N, N): all possible means
unique_means = np.unique(np.round(means, 3)) # select only unique means

debate_by_mean = {}
for bv in betas_to_plot:
    Jt = np.array(J1_two(delta(b_indices[bv])))    # shape (K, K, N, N, 2, 2)

    # LONG picks j1 first, SHORT picks j2 from remaining positions
    long_first  = Jt[:, :, :, :, int(Goal.LONG),  int(Goal.SHORT)]
    # SHORT picks j1 first, LONG picks j2 from remaining positions
    # transpose V axes to align (V_long, V_short) with the judge's frame in memo
    short_first = Jt[:, :, :, :, int(Goal.SHORT), int(Goal.LONG)].transpose(0, 1, 3, 2)

    # Zero out j1==j2: without-replacement means both advocates can't pick same position
    offdiag = (1 - np.eye(K))[:, :, None, None]

    # mat[v_long, v_short] = avg judge posterior P(is_long) over all valid position pairs
    mat = ((long_first + short_first) * offdiag).sum(axis=(0, 1)) / (K * (K - 1) * 2)

    grouped = []
    for m in unique_means:
        mask = np.abs(means - m) < 1e-9
        grouped.append(mat[mask].mean())
    debate_by_mean[bv] = np.array(grouped)

# Plot 
fig = plt.figure(figsize=(12, 4.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
colors = ['#4393c3', '#f4a261', '#e76f51', '#7b1d0e']

# Panel 1: Consultancy
ax1 = fig.add_subplot(gs[0])
ax1.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
ax1.plot(sticks_np, J0_long, 'k-o', lw=1.8, ms=5, label='J0 (literal)')
for bv, col in zip(betas_to_plot[1:], colors[1:]):
    ax1.plot(sticks_np, LONG_J1[bv], '-o', lw=1.8, ms=5, color=col,
             label=f'J1 beta={bv}')
ax1.set_xlabel('stick value u')
ax1.set_ylabel('P(long | u revealed)')
ax1.set_title('Consultancy (single LONG speaker)\nliteral J0 vs pragmatic J1')
ax1.legend(fontsize=8, loc='upper left')
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0.05, 0.95)

# Panel 2: Debate
ax2 = fig.add_subplot(gs[1])
ax2.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
ax2.axvline(0.5, color='k', lw=0.8, ls=':', alpha=0.4)
ax2.plot(unique_means, debate_by_mean[0], 'k-o', lw=1.8, ms=5, label='literal (beta=0)')
for bv, col in zip(betas_to_plot[1:], colors[1:]):
    ax2.plot(unique_means, debate_by_mean[bv], '-o', lw=1.8, ms=3, color=col,
             label=f'J1_two beta={bv}')
ax2.set_xlabel('mean reveal (V_long + V_short) / 2')
ax2.set_ylabel('P(long | V_long, V_short)')
ax2.set_title('Debate (LONG vs SHORT)\nliteral vs pragmatic J1_two')
ax2.legend(fontsize=8, loc='upper left')
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(0.05, 0.95)

plt.savefig('figure3.pdf', bbox_inches='tight')
plt.savefig('figure3.png', bbox_inches='tight', dpi=150)
print('Saved figure3.pdf / .png')
