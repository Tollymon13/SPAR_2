from memo import memo
from enum import IntEnum

import jax
import jax.numpy as np
from functools import partial

from jax.scipy.stats.bernoulli import pmf as ber
from jax.scipy.stats.beta import pdf as beta

import itertools, random

from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import scipy.stats
import statsmodels.api as sm

N = 2
Ph0s = np.array([2/5] * N)
Ph1s = np.array([3/5] * N)
Data = np.arange(2 ** N)
Obs = np.arange(N)
Val = np.array([0, 1])
obs_val_space = np.array(list(itertools.product(Obs, Val)))

def get_datum(d, i):
    return (d >> i) & 1

def p_data_given_h(d, h):
    return np.array([
        ber(get_datum(d, i), np.where(h, Ph1s[i], Ph0s[i]))
        for i in range(N)
    ]).prod()


class H(IntEnum):
    H0 = 0
    H1 = 1

P_MAX = 100
P = np.arange(P_MAX + 1)

class Chi(IntEnum):
    FAIR  = 0
    SYCO  = 1


@memo
def world_model[h: H, d: Data]():
    world: knows(h)
    world: chooses(d in Data, wpp=p_data_given_h(d, h))
    return Pr[world.d == d]

@memo
def bot[p_chi: P, h_human: H, h_world: H, d: Data, obs: Obs, val: Val](prior: ..., level, honest, uniform):
    bot: knows(p_chi, h_human, h_world, d)
    bot: given(chi in Chi, wpp=ber(chi, p_chi / {P_MAX}))
    bot: wants(goal=
        1                       if chi == {Chi.FAIR} else
        (1 if uniform else Pr[human.h_ == h_human]) if chi == {Chi.SYCO} else 1
    )
    bot: chooses(obs in Obs, to_maximize=EU[goal])
    bot: chooses(val in Val, to_maximize=1 * (get_datum(d, obs) == val) if (chi == {Chi.FAIR} or honest) else EU[goal])
    bot: thinks[
        human: knows(h_human, obs, val),
        human: guesses(
            h_ in H, p_chi_ in P,
            wpp=human[h_human, obs, val, h_, p_chi_](prior, level, honest, uniform)
        )
    ]
    return Pr[bot.obs == obs, bot.val == val]

@memo
def human[h: H, obs: Obs, val: Val, _h: H, _p_chi: P](prior: ..., level, honest, uniform):
    human: knows(h)
    human: thinks[
        world: chooses(h in H, p_chi in P, wpp=array_index(prior, h, p_chi)),
        world: chooses(d in Data, wpp=world_model[h, d]()),
        bot: knows(h, world.h, world.p_chi, world.d),
        bot: chooses(
            obs in Obs, val in Val,
            wpp=bot[world.p_chi, h, world.h, world.d, obs, val](prior, level - 1, honest, uniform)
            if level > 0 else 1 * (get_datum(world.d, obs) == val)
        )
    ]
    human: observes [bot.obs] is obs
    human: observes [bot.val] is val
    human: knows(_h, _p_chi)
    return human[Pr[world.h == _h, world.p_chi == _p_chi]]

@memo
def do_sample_uniformly[_h: H](prior: ..., honest):
    return 1.0 + (_h * 0)

@memo
def do_sample_from_prior[_h: H](prior: ..., honest):
    human: thinks[
        world: chooses(h in H, p_chi in P, wpp=array_index(prior, h, p_chi))
    ]
    human: chooses(h in H, wpp=Pr[h == world.h])
    return Pr[human.h == _h]

@memo
def ur_prior[h: H, p_chi: P](p=0.5, prior_syco=1, prior_fair=1):
    human: chooses(h in H, wpp=ber(h, p))
    human: chooses(p_chi in P, wpp=beta(p_chi / {P_MAX}, prior_syco, prior_fair))
    return Pr[human.h == h, human.p_chi == p_chi]


@partial(jax.jit, static_argnames=(
    'time_horizon',
    'human_level',
    'bot_level',
    'human_policy',
    'num_sims',
    'honest',
    'uniform',
    'prior_uniform'
))
def run_sim_jit(
    p_true=0.5,
    p_chi=90,
    time_horizon=50,
    human_level=1,
    bot_level=0,
    human_policy=do_sample_from_prior,
    num_sims=100,
    honest=True,
    uniform=False,
    prior_uniform=True
):
    h_world = H.H1

    def step(carry, t):
        prior, key = carry
        key, key_user, key_world, key_bot = jax.random.split(key, num=4)

        # human ventures opinion
        h_human = jax.random.choice(key_user, np.array(H), p=human_policy(prior, honest))

        # world generates data
        d = jax.random.choice(key_world, np.array(Data), p=world_model()[h_world])

        # bot responds
        obs, val = jax.random.choice(
            key_bot, obs_val_space,
            p=bot(prior=prior, level=bot_level, honest=honest, uniform=uniform)[p_chi, h_human, h_world, d].reshape(-1)
        )

        # human updates belief
        prior = human(prior=prior, level=human_level, honest=honest, uniform=uniform)[h_human, obs, val]

        return (prior, key), prior

    if prior_uniform:
        prior_syco = 1
        prior_fair = 1
    else:
        prior_syco = time_horizon * p_chi/P_MAX + 1
        prior_fair = time_horizon * (1 - p_chi/P_MAX) + 1
    prior = ur_prior(p_true, prior_syco, prior_fair)

    def run_one_sim(seed):
        return jax.lax.scan(f=step, init=(prior, jax.random.key(seed)), length=time_horizon)[1]

    return jax.lax.map(run_one_sim, np.arange(num_sims), batch_size=1000)





Ps_TESTED = P[::10]

if __name__ == '__main__':
    for title, (human_level, honest, uniform) in [
        ("Effectiveness of a sycophantic bot\nagainst a sycophancy-naive user", (0, False, 'prior')),
        ("Effectiveness of the factual sycophant\nagainst a sycophancy-naive user", (0, True, 'prior')),
        ("Effectiveness of the (possibly-fabricating) sycophant\nagainst a sycophancy-aware user", (1, False, 'prior')),
        ("Effectiveness of the factual sycophant\nagainst a sycophancy-aware user", (1, True, 'prior')),
        ("Effectiveness of the fabricating sycophant\nagainst the uniform user", (0, False, 'uniform')),
        ("Effectiveness of the fabricating sycophant\nagainst the uniform aware user", (1, False, 'uniform'))
    ]:
        fname = f'z-{human_level}-{"factual" if honest else "fabricating"}-{uniform}'
        print(fname)
        z = []
        for pi in Ps_TESTED:
            outs = run_sim_jit(
                p_chi=pi,
                human_level=human_level,
                num_sims=10_000,
                time_horizon=100,
                honest=honest,
                uniform=uniform == 'uniform',
                human_policy=do_sample_from_prior
            ).block_until_ready()
            z.append(outs)
        np.save(fname, z)
        print("Done.")
