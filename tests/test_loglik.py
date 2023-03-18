import timeit
from copy import deepcopy
from itertools import product

import demes
import jax
import jax.numpy as jnp
import moments
import numpy as np
import pytest
from jax import jit, value_and_grad
jax.config.update('jax_platform_name', 'cpu')

from momi3.event import ETBuilder
from momi3.MOMI import Momi
from momi3.Params import Params

from io import StringIO


def gutenkunst_no_mig():
    return StringIO('''
description: The Gutenkunst et al. (2009) OOA model.
doi:
- https://doi.org/10.1371/journal.pgen.1000695
time_units: years
generation_time: 25

demes:
- name: ancestral
  description: Equilibrium/root population
  epochs:
  - {end_time: 220e3, start_size: 7300}
- name: AMH
  description: Anatomically modern humans
  ancestors: [ancestral]
  epochs:
  - {end_time: 140e3, start_size: 12300}
- name: OOA
  description: Bottleneck out-of-Africa population
  ancestors: [AMH]
  epochs:
  - {end_time: 21.2e3, start_size: 2100}
- name: YRI
  description: Yoruba in Ibadan, Nigeria
  ancestors: [AMH]
  epochs:
  - start_size: 12300
- name: CHB
  description: Han Chinese in Beijing, China
  ancestors: [OOA]
  epochs:
  - {start_size: 510, end_size: 54090}
''')


def update(params, path, val):
    d = params
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = val


def test_pulse_error():
    # Load demes graph
    demo = demes.load("tests/yaml_files/TwoDemes.yml")
    sampled_demes = ["A", "B"]
    sample_sizes = [5, 10]
    # Construct momi
    momi = Momi(
        demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes, jitted=True
    )
    params = Params(momi)
    params.set_train("eta_1", True)
    params.set_train("tau_1", True)
    jsfs = momi.simulate(10, seed=108)
    momi.loglik(params, jsfs)


def test_gutenkunst_grad():
    n = 5
    demo = demes.load("tests/yaml_files/gutenkunst_ooa.yml")
    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = 3 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    jsfs = momi.simulate(10, seed=108)
    params = momi._default_params
    params.set_train_all_rhos(True)
    params.set_train_all_etas(True)

    (v, g), compilation_time, runtime = momi._time_loglik_with_gradient(
        params, jsfs, batch_size=15, repeat=5
    )
    print(compilation_time)
    print(runtime)


def test_gutenkunst_nan():
    n = 10
    demo = demes.load("tests/yaml_files/gutenkunst_ooa.yml")
    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = 3 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)

    with jax.debug_nans(True):
        momi.sfs_entry(dict(zip(sampled_demes, [0, 0, 1])))


def test_gutenkunst():
    demo = demes.load("tests/yaml_files/gutenkunst_ooa.yml")
    demo = demo.in_generations()
    num_samples = {"YRI": 6, "CEU": 6, "CHB": 6}
    T = ETBuilder(demo, num_samples)
    params = demo.asdict()
    X = {p: np.eye(num_samples[p] + 1)[i] for i, p in enumerate(num_samples)}
    for d in demo.demes:
        if d.name not in X:
            X[d.name] = np.ones(1)

    def f(d, X, auxd):
        p0 = deepcopy(params)
        for path, val in d.items():
            update(p0, path, val)
        return T.execute(p0, X, auxd)

    d = {("demes", 0, "epochs", 0, "start_size"): 1.0}
    jit(value_and_grad(f))(d, X, T.auxd)


def test_gutenkunst_2_pop_full_sfs(n):
    sampled_demes = ["YRI", "CHB"]
    sample_sizes = 2 * [n]

    no_mig_demo = demes.load(gutenkunst_no_mig())
    esfs_momi3_no_mig = Momi(no_mig_demo, sampled_demes, sample_sizes).sfs_spectrum()

    mig_demo = demes.Builder.fromdict(no_mig_demo.asdict())
    mig_demo.add_migration(demes=['CHB', 'YRI'], rate=0.)
    mig_demo = mig_demo.resolve()

    esfs_momi3_mig_0 = Momi(mig_demo, sampled_demes, sample_sizes).sfs_spectrum()

    esfs_moments = moments.Spectrum.from_demes(
        mig_demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
    ) * mig_demo.demes[0].epochs[0].start_size * 4

    l1 = lambda x, y: np.abs(x - y).mean()
    print(f'l1(momi3 mig=0, moments mig=0): {l1(esfs_momi3_mig_0, esfs_moments)}')
    print(f'l1(momi3 nomig, moments mig=0): {l1(esfs_momi3_no_mig, esfs_moments)}')


@pytest.mark.parametrize("n,size", [(1, 1), (2, 2), (10, 10)])
def test_gutenkunst_vmap(n, size):
    demo = demes.load("tests/yaml_files/gutenkunst_ooa.yml")
    demo = demo.in_generations()
    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = 3 * [n]
    num_samples = dict(zip(sampled_demes, sample_sizes))

    bs = [jnp.arange(num_samples[pop] + 1) for pop in num_samples]
    mutant_sizes = jnp.array(list(product(*bs)))
    num_deriveds = {}
    for i, pop in enumerate(num_samples):
        num_deriveds[pop] = mutant_sizes[:, i]
    for pop in num_deriveds:
        num_deriveds[pop] = num_deriveds[pop][:size]

    T = ETBuilder(demo, num_samples)
    demo_dict = demo.asdict()
    params = Params(demo_dict=demo_dict, T=T)
    leaves = T._leaves

    def moments_esfs():
        esfs = moments.Spectrum.from_demes(
            demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
        )
        esfs = np.array(esfs).flatten()[1:-1]
        return esfs.sum()

    # nit = 25  # number of iterations for timing f
    # t = timeit(lambda: moments_esfs(), number=nit) / nit
    # print(f"moments: {t:.2g} seconds")

    @jit
    def f(train_dict, nuisance_dict, num_derived, auxd):
        X = {}
        for pop in leaves:
            # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
            ns = num_samples.get(pop, 0)
            d = num_derived.get(pop, 0)
            X[pop] = jax.nn.one_hot(jnp.array([d]), ns + 1)[0]
        theta_dict = deepcopy(nuisance_dict)
        theta_dict.update(train_dict)
        dd0 = deepcopy(demo_dict)
        for paths, val in theta_dict.items():
            for path in paths:
                update(dd0, path, val)
        return T.execute(dd0, X, auxd)

    @jit
    def vmap_sum_f(train_dict, nuisance_dict, num_derived, auxd):
        sfs = jax.vmap(f, (None, None, 0, None))(
            train_dict, nuisance_dict, num_derived, auxd
        )
        return sfs.sum()

    jvag = jit(value_and_grad(vmap_sum_f))

    params.set_train_all_etas(True)  # Take grad for size params
    train_dict = params._theta_train_dict
    nuisance_dict = params._theta_nuisance_dict

    times = []
    for j in range(size):
        ts = timeit.default_timer()
        f(
            train_dict,
            nuisance_dict,
            {i: num_deriveds[i][j] for i in num_samples},
            T.auxd,
        )
        te = timeit.default_timer()
        times.append(te - ts)
    t = np.median(times)
    print(f"momi3 loglik single pass: {t:.2g} seconds")

    nit = 10
    times = timeit.repeat(
        lambda: vmap_sum_f(train_dict, nuisance_dict, num_deriveds, T.auxd),
        number=1,
        repeat=nit,
    )
    t = np.median(times)
    print(f"momi3 loglik vmap of {size=}: {t:.2g} seconds")

    df = jit(value_and_grad(f))
    times = []
    for j in range(size):
        ts = timeit.default_timer()
        df(
            train_dict,
            nuisance_dict,
            {i: num_deriveds[i][j] for i in num_samples},
            T.auxd,
        )
        te = timeit.default_timer()
        times.append(te - ts)
    t = np.median(times)
    print(f"momi3 grad of loglik single pass: {t:.2g} seconds")

    nit = 10
    times = timeit.repeat(
        lambda: jvag(train_dict, nuisance_dict, num_deriveds, T.auxd),
        number=1,
        repeat=nit,
    )
    t = np.median(times)
    print(f"momi3 grad of loglik vmap of {size=}: {t:.2g} seconds")


if __name__ == "__main__":
    test_gutenkunst_2_pop_full_sfs(1)
