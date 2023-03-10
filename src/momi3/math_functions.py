import jax
import jax.numpy as jnp
from jax import custom_jvp, lax, vmap
from jax._src.scipy.special import _expn1, _expn2
from jax.numpy import exp
from jax.scipy.special import xlog1py, xlogy
from scipy.special import betaln, gammaln


@custom_jvp
@jnp.vectorize
@jax.jit
def exp1(x):
    x = jnp.array(x, dtype="f")
    is_x_small = x < 1.0
    x_safe = jnp.where(is_x_small, 100.0, x)
    e1 = _expn1(1, x)
    e2 = _expn2(1, x_safe)
    ret = jnp.where(is_x_small, e1, e2)
    return ret


@exp1.defjvp
@jax.jit
def exp1_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return exp1(x), lax.mul(lax.neg(x_dot), exp1(x))


def _expi_neg(x):
    # expi for x < 0
    return -exp1(-x)


def _expi_pos_small(x):
    # expi for 0. < x < 7.1
    gamma = 0.5772157
    ret = gamma + jnp.log(x)
    p = 1 / jnp.array(
        [439084800, 36288000, 3265920, 322560, 35280, 4320, 600, 96, 18, 4, 1, jnp.inf]
    )
    return ret + jnp.polyval(p, x)


def _expi_pos_large(x):
    # expi for x > 7.1
    ret = jnp.exp(x) / x
    p = jnp.array([3628800, 362880, 40320, 5040, 720, 120, 24, 6, 2, 1, 1])
    return ret * jnp.polyval(p, 1 / x)


@custom_jvp
@jnp.vectorize
@jax.jit
def aexpi(x):
    # approximate expi
    return lax.cond(
        x < 0,
        _expi_neg,
        lambda x: lax.cond(x < 7.1, _expi_pos_small, _expi_pos_large, x),
        x,
    )


@aexpi.defjvp
def aexpi_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return aexpi(x), jnp.exp(x) / x * x_dot


def logFactorial(x):
    return gammaln(x + 1)


def logBinom(n, k):
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k)


def expm1d_series(x):
    p = 1 / jnp.array([3628800, 362880, 40320, 5040, 720, 120, 24, 6, 2, 1])
    return jnp.polyval(p, x)


def expm1d_naive(x):
    # used by exp_integralEGPS
    return jnp.expm1(x) / x


def TEi_series(x):
    p = jnp.array([3628800, -362880, 40320, -5040, 720, -120, 24, -6, 2, -1, 1])
    return jnp.polyval(p, x)


def TEi_naive(x):
    # used by exp_integralEGPS
    x = jnp.array(x, dtype="f")
    y = 1.0 / x
    return -aexpi(-y) * jnp.exp(y) / x


def expm1d(x):
    # used by exp_integralEGPS
    is_x_small = jnp.isclose(x, 0.0)
    x_safe = jnp.where(is_x_small, 1.0, x)
    NV = expm1d_naive(x_safe)
    TS = expm1d_series(x)
    return jnp.where(is_x_small, TS, NV)


def TEi(x):
    # used by exp_integralEGPS
    is_x_small = jnp.abs(x) < 0.015
    x_safe = jnp.where(is_x_small, 1.0, x)
    NV = TEi_naive(x_safe)
    TS = TEi_series(x)
    return jnp.where(is_x_small, TS, NV)


def admix_inner_loop(nw, x1, x2, xw, m1, q):
    # used by admix_outer_loop
    B = log_binom_pmf(m1, nw, 1 - q)
    j1s = jnp.arange(nw + 1)
    j2s = xw - j1s
    m2 = nw - m1

    H1 = log_hypergeom(j1s, nw, m1, x1)
    H2 = log_hypergeom(j2s, nw, m2, x2)
    H = H1 + H2
    return jnp.exp(B + H).sum()


def admix_outer_loop(lik, x1, x2, q):
    """
    Returns the outer loop for admix lemma
    lik: likelihood of admixed population
    x1: Number of mutant lineages in parent1
    x2: Number of mutant lineages in parent2
    q: Admix proportion of parent2
    """
    nw = lik.shape[0] - 1
    m1s = jnp.arange(nw + 1)
    xws = jnp.arange(nw + 1)

    m1s_slice = 6 * [None]
    m1s_slice[4] = 0
    xws_slice = 6 * [None]
    xws_slice[3] = 0

    f_inner_sum = jax.vmap(jax.vmap(admix_inner_loop, m1s_slice), xws_slice)
    inner_sum = f_inner_sum(nw, x1, x2, xws, m1s, q)
    return jnp.einsum("ab,a...->...", inner_sum, lik)


def exp_integral(a, tau, j):
    r"""
    Returns exponential integral of coalescent rate for constant pop size: \int_0^tau exp(-R(t))
    a: constant coalescent rate
    tau: truncation time
    j: rate coefficient
    """
    a = a * j
    tauinf = jnp.isinf(tau)
    tau_safe = jnp.where(tauinf, 1.0, tau)
    ret = jnp.where(tauinf, 1 / a, expm1d(-a * tau_safe) * tau_safe)
    return ret


def exp_integralEGPS(g, a, tau, j):
    r"""
    Returns the exponential integral of coalescent rate for exponential pop size: \int_0^tau exp(-R(t))
    g: growth rate
    a: coalescent rate at the bottom (backwards in time)
    tau: truncation time (must be less than infinity)
    j: rate coefficient
    """
    pow0 = 1 / a / j
    pow1 = g * tau
    ret = -TEi(pow0 * g / exp(pow1))
    ret = ret * exp(-expm1d(pow1) * tau / pow0 - pow1)
    ret = ret + TEi(pow0 * g)
    ret = ret * pow0
    return ret


def log_hypergeom(k, M, n, N):
    """
    Returns the log of hyper geometric coefficient
    k: number of selected Type I objects
    M: total number of objects
    n: total number of Type I objects
    N: Number of draws without replacement from the total population
    """
    # https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_discrete_distns.py
    tot, good = M, n
    bad = tot - good
    result = (
        betaln(good + 1, 1)
        + betaln(bad + 1, 1)
        + betaln(tot - N + 1, N + 1)
        - betaln(k + 1, good - k + 1)
        - betaln(N - k + 1, bad - N + k + 1)
        - betaln(tot + 1, 1)
    )
    return result


def log_binom_pmf(k, n, p):
    """
    Returns the log of binomial pmf
    k: number of success
    n: sample size
    p: success probability
    """
    p = jnp.clip(p, a_min=1e-5, a_max=1 - 1e-5)
    return logBinom(n, k) + xlogy(k, p) + xlog1py(n - k, -p)


def convolve_sum(A, B):
    "C[j,k,l+m] += A[i,j,l,n] * B[i,k,m,n]"
    k = B.shape[2]

    def f1(aj, bk):
        return lax.conv_general_dilated(
            aj[None, None],
            bk[
                None,
                None,
                ::-1,
            ],
            (1, 1),
            ((k - 1, k - 1), (0, 0)),
        ).squeeze()

    f2 = vmap(f1, (None, 0))
    f3 = vmap(f2, (0, None))
    return f3(A, B)
