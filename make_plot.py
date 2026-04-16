from matplotlib import pyplot as plt
from delusion import *

@jax.jit
def get_spiralers(z):
    z_ = z.sum(axis=-1)[..., H.H0]
    z_ = (z_ > 0.99).any(axis=-1)
    return z_

def p_to_stars(p):
    if p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def get_confidence_intervals(counts, props, nobs):
    ci_abs = sm.stats.proportion_confint(counts, nobs)
    return np.abs(props - np.array(ci_abs))

def plot_results(z, bar=True, label='', dodge=0, color='k', ylabel=False, alpha=1, ls='-', stars=True):
    z_ = get_spiralers(z)
    nobs = z_.shape[-1]
    counts = z_.sum(axis=-1)
    props = z_.mean(axis=-1)
    plt.errorbar(
        Ps_TESTED / P_MAX + dodge,
        props,
        yerr=get_confidence_intervals(counts, props, nobs),
        capsize=3,
        label=label,
        c=color, alpha=alpha, ls=ls
    )
    if stars:
        plt.axhline(props[0], ls=':', c=color, alpha=0.5)
    plt.xlabel('Rate of sycophantic/hallucinated responses (π)')
    if ylabel:
        plt.ylabel('Rate of catastrophic\ndelusional spiraling')




plt.figure(figsize=(6, 9))
i, j, n = 4, 1, 0

plt.subplot(i, j, n := n + 1)
plt.title('(A) Naive user, hallucinating bot')

z = np.load('z-0-fabricating-prior.npy')
plot_results(z, bar=False, color='r', ylabel=True, label='Sycophantic')
plt.ylim(-0.01, 0.61)

z = np.load('z-0-fabricating-uniform.npy')
plot_results(z, bar=False, color='r', stars=False, ls='--', label='Non-sycophantic')

plt.legend()



plt.subplot(i, j, n := n + 1)
plt.title('(B) Naive user, factual bot')

z = np.load('z-0-factual-prior.npy')
plot_results(z, bar=False, ylabel=True, color='b')
plt.ylim(-0.01, 0.61)




plt.subplot(i, j, n := n + 1)
plt.title('(C) Informed user, hallucinating bot')

z = np.load('z-1-fabricating-prior.npy')
plot_results(z, bar=False, color='g', ylabel=True, label='Sycophantic')
plt.ylim(-0.001, 0.017)

z = np.load('z-1-fabricating-uniform.npy')
plot_results(z, bar=False, color='g', stars=False, ls='--', label='Non-sycophantic')

plt.legend()




plt.subplot(i, j, n := n + 1)
plt.title('(D) Informed user, factual bot')

z = np.load('z-1-factual-prior.npy')
plot_results(z, bar=False, ylabel=True, color='m')
plt.ylim(-0.001, 0.017)


plt.tight_layout()
plt.savefig('extensions.pdf')
