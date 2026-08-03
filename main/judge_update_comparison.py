import sys
sys.path.insert(0, 'main')

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from barnett_rsa import J0_long, J1, J1_two, B, B_vals, Goal, STICKS, N, K

sticks_np = np.array(STICKS)

consult_betas = [0, 2, 5, 10]       # betas for consultancy line plot
debate_betas  = [0, 5, 10, 20]      # betas for debate heatmaps
all_betas = sorted(set(consult_betas + debate_betas))
b_indices = {bv: int(np.where(np.array(B_vals) == float(bv))[0][0]) for bv in all_betas}

# delta function representing beta value (i.e. judge knows beta exactly under Barnett's eq. 7)
def delta(b_idx):
    d = np.zeros(len(B)); d[b_idx] = 1.0
    return jnp.array(d)

# Panel 1: Consultancy (single LONG speaker, Barnett Eq. 7)
LONG_J1 = {}
for bv in consult_betas:
    J1_table = np.array(J1(delta(b_indices[bv])))  # shape (N, 2): stick values x goals
    LONG_J1[bv] = J1_table[:, int(Goal.LONG)]      # P(long | v, LONG speaker, beta=bv)

# Panel 2: Debate (LONG vs SHORT, symmetric over all position pairs)
# Compute full belief surface mat[v_long, v_short] for each beta — shown as 2x2 heatmaps
debate_mats = {}
for bv in debate_betas:
    Jt = np.array(J1_two(delta(b_indices[bv])))    # shape (K, K, N, N, 2, 2)

    # LONG picks j1 first, SHORT picks j2 from remaining positions
    long_first  = Jt[:, :, :, :, int(Goal.LONG),  int(Goal.SHORT)]
    # SHORT picks j1 first, LONG picks j2 from remaining positions
    # transpose V axes to align (V_long, V_short) with the judge's frame in memo
    short_first = Jt[:, :, :, :, int(Goal.SHORT), int(Goal.LONG)].transpose(0, 1, 3, 2)

    # Zero out j1==j2: without-replacement means both advocates can't pick same position
    offdiag = (1 - np.eye(K))[:, :, None, None]

    # mat[v_long_idx, v_short_idx] = avg judge posterior P(is_long) over all valid position pairs
    debate_mats[bv] = ((long_first + short_first) * offdiag).sum(axis=(0, 1)) / (K * (K - 1) * 2)

# --- Figure 1: Consultancy ---
colors = ['#4393c3', '#f4a261', '#e76f51', '#7b1d0e']

fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.4)
ax1.plot(sticks_np, J0_long, 'k-o', lw=1.8, ms=5, label='J0 (literal)')
for bv, col in zip(consult_betas[1:], colors[1:]):
    ax1.plot(sticks_np, LONG_J1[bv], '-o', lw=1.8, ms=5, color=col,
             label=f'J1 β={bv}')
ax1.set_xlabel('stick value u')
ax1.set_ylabel('P(long | u revealed)')
ax1.set_title('Consultancy (single LONG speaker)\nliteral J0 vs pragmatic J1')
ax1.legend(fontsize=8, loc='upper left')
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(0.05, 0.95)

fig1.tight_layout()
fig1.savefig('figure3_consultancy.pdf', bbox_inches='tight')
fig1.savefig('figure3_consultancy.png', bbox_inches='tight', dpi=150)
print('Saved figure3_consultancy.pdf / .png')

# --- Figure 2: Debate heatmaps (2x2, one per beta) ---
# Each heatmap shows mat[v_long, v_short] = P(long | reveals) for all 9x9 reveal combinations
# x-axis: V_long (LONG speaker's reveal); y-axis: V_short (SHORT speaker's reveal)
# Bottom-right (high V_long, low V_short): LONG dominates → red
# Top-left (low V_long, high V_short): SHORT dominates → blue
# Higher beta: sharper contrast (more decisive judge); upper-left corner turns white at high beta
# because LONG showing 0.1 (world max ≈ 0.1) and SHORT showing 0.9 (world min ≈ 0.9) is impossible
tick_labels  = [f'{s:.1f}' for s in sticks_np]
beta_labels  = ['β=0 (literal)', 'β=5', 'β=10', 'β=20']

fig2, axes = plt.subplots(2, 2, figsize=(10, 9))
for idx, (bv, blabel) in enumerate(zip(debate_betas, beta_labels)):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    mat = debate_mats[bv]
    # mat[i,j] = P(long | v_long=STICKS[i], v_short=STICKS[j])
    # transpose so columns=v_long (x-axis), rows=v_short (y-axis); origin='lower' puts low values at bottom
    im = ax.imshow(mat.T, origin='lower', vmin=0, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(N)); ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(N)); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel('V_long (LONG reveal)', fontsize=9)
    ax.set_ylabel('V_short (SHORT reveal)', fontsize=9)
    ax.set_title(blabel, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(long)')

fig2.suptitle('Debate (LONG vs SHORT): judge belief P(long | V_long, V_short)', y=1.01)
fig2.tight_layout()
fig2.savefig('figure3_debate.pdf', bbox_inches='tight')
fig2.savefig('figure3_debate.png', bbox_inches='tight', dpi=150)
print('Saved figure3_debate.pdf / .png')
