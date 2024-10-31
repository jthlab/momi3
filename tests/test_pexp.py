import numpy as np
import pytest
from scipy.integrate import quad

from momi3.pexp import PExp


@pytest.fixture
def rpexp(rng):
    N0 = rng.random(10)
    N1 = rng.random(10)
    t = np.cumsum(rng.random(11))
    return PExp(N0, N1, t)


def test_pexp_call(rpexp, rng):
    t = rpexp.t
    N0 = rpexp.N0
    N1 = rpexp.N1
    for tt in rng.uniform(t[0], t[-1], 10):
        i = np.searchsorted(t, tt) - 1
        dt = (t[i + 1] - tt) / (t[i + 1] - t[i])
        r = np.log(N0[i] / N1[i])
        N = N1[i] * np.exp(r * dt)
        np.testing.assert_allclose(rpexp(tt), N)


def test_pexp_R0():
    "test using quadrature that PExp(N0, N1, r).R(t) is the integral of PExp(N0, N1, r)(t) from 0 to t"
    pe = PExp(
        t=np.array([0.0, 1.0, 2.0]), N0=np.array([1.0, 1.0]), N1=np.array([2, 2.0])
    )
    np.testing.assert_allclose(pe(0.0), 1.0)
    np.testing.assert_allclose(pe(1.0), 2.0)
    np.testing.assert_allclose(pe.R(0.0), 0.0)
    np.testing.assert_allclose(pe.R(0.5), (2 - np.sqrt(2)) / (4 * np.log(2)))
    np.testing.assert_allclose(pe.R(1.0), 1 / (4 * np.log(2)))
    np.testing.assert_allclose(pe.R(2.0), 2 / (4 * np.log(2)))
    np.testing.assert_allclose(pe.R(1.5), (1 + 2 - np.sqrt(2)) / (4 * np.log(2)))


def test_pexp_ab(rpexp, rng):
    "test that PExp(N0, N1, r).a and PExp(N0, N1, r).b are correct"
    for tt in rng.uniform(rpexp.t[0], rpexp.t[-1], 10):
        i = np.searchsorted(rpexp.t, tt) - 1
        assert rpexp.t[i] <= tt < rpexp.t[i + 1]
        np.testing.assert_allclose(
            1 / 2 / rpexp(tt), rpexp.a[i] * np.exp(-rpexp.b[i] * (rpexp.t[i + 1] - tt))
        )


def test_pexp_R(rpexp, rng):
    "test using quadrature that PExp(N0, N1, r).R(t) is the integral of PExp(N0, N1, r)(t) from 0 to t"
    t = rpexp.t
    rpexp.N0
    rpexp.N1
    for tt in rng.uniform(t[0], t[-1], 10):
        q, err = quad(lambda x: 1 / (2 * rpexp(x)), t[0], tt, points=t)
        np.testing.assert_allclose(rpexp.R(tt), q, atol=err)


def test_pexp_exp_integral(rng):
    "test that PExp(N0, N1, r).exp_integral(t0, t1) is the integral of exp(-r [R(t) - R(t0)]) from t0 to t1"
    N0 = rng.random(10)
    N1 = rng.random(10)
    t = np.append(np.cumsum(rng.random(10)), np.inf)
    N0[-1] = N1[-1]  # make the last interval constant
    pe = PExp(N0, N1, t)

    # test also the case where t1 is infinite
    q, err = quad(lambda x: np.exp(-(pe.R(x) - pe.R(t[-2]))), t[-2], np.inf)
    np.testing.assert_allclose(pe.exp_integral(t[-2], np.inf), q, rtol=1e-5)

    for t0, t1 in rng.uniform(t[0], 2 * t[-2], (10, 2)):
        t0, t1 = sorted([t0, t1])
        q, err = quad(lambda x: np.exp(-(pe.R(x) - pe.R(t0))), t0, t1, points=t)
        np.testing.assert_allclose(pe.exp_integral(t0, t1), q, rtol=1e-5)


def test_pexp_exp_integral_const(rng):
    N0 = rng.random(1)
    t = np.array([0.0, np.inf])
    for u in rng.uniform(0.0, 10.0, 10):
        pe = PExp(N0, N0, t)
        np.testing.assert_allclose(pe.R(u), u / 2 / N0, rtol=1e-5)
