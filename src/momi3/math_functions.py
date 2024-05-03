import jax
import jax.numpy as jnp
import quadax
from jax import lax, vmap
from scipy.special import betaln


def binom_pmf_safe(k, n, p):
    p_safe = jnp.where(jnp.isclose(p, 0.0) | jnp.isclose(p, 1.0), 0.5, p)
    return jnp.select(
        [jnp.isclose(p, 0.0), jnp.isclose(p, 1.0)],
        [(k == 0).astype(float), (k == n).astype(float)],
        jax.scipy.stats.binom.pmf(k, n, p_safe),
    )


def admix_inner_loop(nw, x1, x2, xw, m1, q):
    # used by admix_outer_loop
    B = binom_pmf_safe(m1, nw, 1 - q)
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


def expm1d(x):
    "(exp(x) - 1)/x"
    x_small = abs(x) < 1e-6
    x_safe = jnp.where(x_small, 1.0, x)
    return jnp.where(x_small, 1 + x / 2, jnp.expm1(x_safe) / x_safe)


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
    c = tau_safe * expm1d(-a * tau_safe)
    return jnp.where(tauinf, 1 / a, c)


def exp_integralEGPS(g, a, tau, j):
    r"""
    Returns the exponential integral of coalescent rate for exponential pop size: \int_0^tau exp(-R(t))
    g: growth rate
    a: coalescent rate at the bottom (backwards in time)
    tau: truncation time (must be less than infinity)
    j: rate coefficient
    """
    g = -g

    def R(t):
        g_small = abs(g) < 1e-6
        g_safe = jnp.where(g_small, 1.0, g)
        r1 = -jnp.expm1(-g_safe * t) / g_safe
        # r2 = taylor series expansion of r1 about g=0
        r2 = t * (1 - g * t / 2)
        return a * jnp.where(g_small, r2, r1)

    y, info = quadax.quadgk(lambda x: jnp.exp(-R(x)), [0.0, tau])
    return y


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


def softplus_inverse(tx):
    return jnp.where(tx > 50, tx, jnp.log(jnp.expm1(tx)))
