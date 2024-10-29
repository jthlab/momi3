import numpy as np
from scipy.integrate import quad

from momi3.pexp import PExp


def test_pexp_R(rng):
    "test using quadrature that PExp(N0, N1, r).R(t) is the integral of PExp(N0, N1, r)(t) from 0 to t"
    N0 = rng.random(10)
    N1 = rng.random(10)
    t = np.cumsum(rng.random(11))
    pe = PExp(N0, N1, t)
    for tt in rng.uniform(t[0], t[-1], 10):
        q, err = quad(pe, t[0], tt, points=t)
        np.testing.assert_allclose(pe.R(tt), q, atol=err)


def test_pexp_exp_integral(rng):
    "test that PExp(N0, N1, r).exp_integral(t0, t1) is the integral of exp(-r [R(t) - R(t0)]) from t0 to t1"
    N0 = rng.random(10)
    N1 = rng.random(10)
    t = np.append(np.cumsum(rng.random(10)), np.inf)
    N0[-1] = N1[-1]  # make the last interval constant
    pe = PExp(N0, N1, t)

    # test also the case where t1 is infinite
    q, err = quad(lambda x: np.exp(-(pe.R(x) - pe.R(t[-2]))), t[-2], np.inf)
    np.testing.assert_allclose(pe.exp_integral(t[-2], np.inf), q, atol=1e-5)

    for t0, t1 in rng.uniform(t[0], 2 * t[-2], (10, 2)):
        t0, t1 = sorted([t0, t1])
        q, err = quad(lambda x: np.exp(-(pe.R(x) - pe.R(t0))), t0, t1, points=t)
        np.testing.assert_allclose(pe.exp_integral(t0, t1), q, atol=1e-5)


def test_pexp_exp_integral_const(rng):
    N0 = rng.random(1)
    t = np.array([0.0, np.inf])
    for u in rng.uniform(0.0, 10.0, 10):
        pe = PExp(N0, N0, t)
        np.testing.assert_allclose(pe.R(u), u / N0, atol=1e-5)
