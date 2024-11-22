import itertools as it
from collections import Counter
from functools import partial

import jax
import jax.numpy as jnp
import msprime
import numpy as np
import pytest
import stdpopsim

import momi3.momi
from momi3 import Momi3

from .demos import SingleDeme, ThreeDemes, TwoDemes

jax.config.update("jax_enable_x64", True)


def _idfun(x):
    if isinstance(x, stdpopsim.DemographicModel):
        return x.id
    else:
        return "".join(x)


@pytest.mark.parametrize(
    "demo,pops",
    [
        (demo, Counter([pop.name for pop in pops]))
        for demo in stdpopsim.all_demographic_models()
        for pops in it.combinations_with_replacement(demo.populations, 2)
    ],
    ids=_idfun,
)
def test_stdpopsim(demo, pops):
    t0 = 0.0
    t1 = 10.0 * demo.model.debug().epoch_start_time[-1]
    t = np.linspace(t0, t1, 12345)
    model_times = np.array([e.time for e in demo.model.events])
    t = np.sort(np.unique(np.concatenate([t, model_times])))
    m3 = momi3.momi.Momi3(demo.model.to_demes())
    c2, p2 = m3.coalescence_rate_trajectory(t, pops)
    c1, p1 = demo.model.debug().coalescence_rate_trajectory(steps=t, lineages=pops)
    # thle iicr can jump discontinuously at the model times, so the value depends on
    # whether the function is defined to be left- or right-continuous. afaict theres's
    # not really a convention for this so we just ignore the values at the model times
    tm = np.isclose(t[:, None], model_times[None, :]).any(1)
    np.testing.assert_allclose(c1[~tm], c2[~tm], rtol=1e-6, atol=1e-6)
    # FIXME this does not quite match, appears to be due to numerical inaccuracy in msprime
    # np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_iicr_star():
    demo = msprime.Demography()
    pops = "ABCDE"
    for p in pops:
        demo.add_population(name=p, initial_size=1e4)
    demo.add_population(name="anc", initial_size=1e5)
    demo.add_population_split(time=1e3, derived=pops, ancestral="anc")
    m3 = momi3.momi.Momi3(demo.to_demes())
    print(m3.iicr({"A": 2}).constraints)
    t = np.append(np.linspace(0.0, 1.1e4, 12345), 1e3)
    t.sort()
    for d in map(Counter, it.combinations_with_replacement(pops, 2)):
        c, p = m3.coalescence_rate_trajectory(t, d)
        if len(d) == 1:
            np.testing.assert_allclose(c[t < 1e3], 1 / 2 / 1e4)
        else:
            np.testing.assert_allclose(c[t < 1e3], 0.0)
        np.testing.assert_allclose(c[t >= 1e3], 1 / 2 / 1e5)

    # test some other one-off cases
    d = {"A": 2, "B": 1}
    c, p = m3.coalescence_rate_trajectory(t, d)
    np.testing.assert_allclose(c[t < 1e3], 1 / 2 / 1e4)
    np.testing.assert_allclose(c[t >= 1e3], 3 / 2 / 1e5)

    d = {"A": 2, "B": 2}
    c, p = m3.coalescence_rate_trajectory(t, d)
    np.testing.assert_allclose(c[t < 1e3], 2 / 2 / 1e4)
    np.testing.assert_allclose(c[t >= 1e3], 6 / 2 / 1e5)

    d = {"A": 2, "B": 2, "C": 1}
    c, p = m3.coalescence_rate_trajectory(t, d)
    np.testing.assert_allclose(c[t < 1e3], 2 / 2 / 1e4)
    np.testing.assert_allclose(c[t >= 1e3], 10 / 2 / 1e5)


def test_iicr_simple():
    demo, _ = SingleDeme.Constant().base()
    # FIXME: rate at changepoints is not handled correctly because of autodiff
    t = np.linspace(0.0, 1.1e4, 10)
    N = 1e4
    for n in [2, 5, 20]:
        c, p = Momi3(demo).coalescence_rate_trajectory(t, {"A": n})
        np.testing.assert_allclose(p, np.exp(-t * n * (n - 1) / 4 / N))
        np.testing.assert_allclose(c, n * (n - 1) / 4 / N)


def test_iicr_growth():
    demo, _ = SingleDeme.Exponential().base()
    t = jnp.linspace(0, 2.1e4, 20)
    N_t = np.array([demo.demes[0].size_at(tt) for tt in t])
    for n in [2, 5, 20]:
        c, p = Momi3(demo).coal_rate_trajectory(t, {"A": n})
        np.testing.assert_allclose(c, n * (n - 1) / 4 / N_t)


def test_iicr_twopop():
    demo, _ = TwoDemes.Constant().base()
    t = jnp.linspace(0, 2e4, 20)

    for lin in "A", "B":
        d = {lin: 2}
        c, p = Momi3(demo).coalescence_rate_trajectory(t, d)
        deme = next(d for d in demo.demes if d.name == lin)
        N_t = np.array([deme.size_at(tt) for tt in t])
        np.testing.assert_allclose(c, 1 / 2 / N_t)
        np.testing.assert_allclose(p, np.exp(-t / 2 / N_t))


def test_iicr_mig0_vs_msp():
    exp = TwoDemes.Exponential()
    demo1, _ = exp.base()
    demo2, _ = exp.migration(tstart=1e3, tend=0.0, rate=0.0)
    t = np.linspace(0.0, 1.1e4, 12345)
    dd1, dd2 = [msprime.Demography.from_demes(d).debug() for d in (demo1, demo2)]
    for d in [{"A": 2}, {"B": 2}]:
        c1, p1 = dd1.coalescence_rate_trajectory(steps=t, lineages=d)
        # first check that these are equal
        c2, p2 = dd2.coalescence_rate_trajectory(steps=t, lineages=d)
        np.testing.assert_allclose(c1, c2, atol=1e-6)
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        1 / 2 / dd1.population_size_trajectory(t)
        1 / 2 / dd2.population_size_trajectory(t)
        # thin check that the iicr is the same
        for de in demo1, demo2:
            m3 = momi3.momi.Momi3(de)
            c3, p3 = m3.coalescence_rate_trajectory(t, d)
            np.testing.assert_allclose(c1, c3, rtol=1e-4)
            np.testing.assert_allclose(p1, p3, atol=1e-4)


def test_iicr_iwm():
    cons = TwoDemes.Constant()
    demo, _ = cons.migration()
    t = np.linspace(0.0, 1.1 * cons.t, 123456)
    c2, p2 = Momi3(demo).coalescence_rate_trajectory(t, {"A": 1, "B": 1})
    c1, p1 = (
        msprime.Demography.from_demes(demo)
        .debug()
        .coalescence_rate_trajectory(steps=t, lineages={"A": 1, "B": 1})
    )
    np.testing.assert_allclose(c1, c2, rtol=1e-4)
    np.testing.assert_allclose(p1, p2, rtol=1e-4)


@pytest.mark.parametrize(
    "demo_gen,n",
    [
        (
            partial(TwoDemes.Constant(t=1000.0).migration_sym, tstart=1000.0, tend=0.0),
            2,
        ),
        (ThreeDemes.Constant(t1=1000.0, t2=1001.0).three_migrants, 3),
    ],
)
def test_strobeck(demo_gen, n, rng):
    r = jnp.clip(rng.exponential(0.1), 1 / n)
    demo = demo_gen(rate=r)[0]
    np.linspace(0.0, 100.0, 12345)
    m3 = momi3.momi.Momi3(demo)
    for tup in it.combinations_with_replacement("ABC"[:n], 2):
        d = Counter(tup)
        iicr = m3.iicr(d)
        m = n  # within
        if len(d) == 2:
            # between
            m += 1 / 4 / r
        np.testing.assert_allclose(iicr.ET(), 2.0 * m, atol=1e-3)  # = 2 N n


@pytest.mark.parametrize(
    "config", map(Counter, it.combinations_with_replacement("ABC", 2))
)
@pytest.mark.parametrize(
    "demo_tup",
    [
        TwoDemes.Constant().two_pulses(),
        TwoDemes.Constant().pulse(),
        TwoDemes.Constant().base(),
        TwoDemes.Exponential().base(),
        TwoDemes.Constant().migration(),
        TwoDemes.Constant().migration_twophase(),
        TwoDemes.Exponential().migration(),
        TwoDemes.Exponential().migration_sym(),
        ThreeDemes.Constant().base(),
        ThreeDemes.Constant().migration(),
    ],
)
def test_vs_msp(demo_tup, config):
    demo, _ = demo_tup
    if not all(any(d.name == x for d in demo.demes) for x in config):
        pytest.skip("not a valid configuration for this demo")
    t = np.linspace(0.0, 3.1e4, 123456)
    dd = msprime.Demography.from_demes(demo).debug()
    m3 = momi3.momi.Momi3(demo)
    c2, p2 = m3.coalescence_rate_trajectory(t, config)
    c1, p1 = dd.coalescence_rate_trajectory(steps=t, lineages=config)
    mask = p1 > 1e-8
    np.testing.assert_allclose(p1[mask], p2[mask], atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(c1[mask], c2[mask], atol=1e-3, rtol=1e-3)


def test_pulse():
    demo, _ = TwoDemes.Constant().two_pulses()
    t = np.linspace(0.0, 1.1e4, 123456)
    m3 = momi3.momi.Momi3(demo)
    c1, p1 = m3.coalescence_rate_trajectory(t, {"A": 1, "B": 1})
