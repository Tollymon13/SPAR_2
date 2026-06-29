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

# Validation
print("J0(long | v=0.5):", round(float(J0_long[4]), 4))
for bv in betas_to_plot:
    print(f"J1(long | v=0.5, beta={bv}):", round(float(LONG_J1[bv][4]), 4))

