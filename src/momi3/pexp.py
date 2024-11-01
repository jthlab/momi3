from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap


class PExp(NamedTuple):
    """Piecewise exponential rate function.

    This represents the function

        eta(t) = N0[i] (N1[i] / N0[i])^[(t-t[i])/(t[i+1]-t[i])] for t_i <= t < t_{i+1}

    i.e. eta(t[i])=N0[i], eta(t[i+1])=N1[i], and eta(t) is exponential between t[i] and t[i+1].

    Args:
        N0, N1: positive arrays of shape [T] corresponding to the formula shown above.
        t: positive array of shape [T + 1] corresponding to t_i in the formula shown above.
    """

    N0: jnp.ndarray
    N1: jnp.ndarray
    t: jnp.ndarray

    def reverse(self):
        r"Return a new PExp object with time reversed."
        # won't work if self.t[-1] = inf, but we should not ever hit this case
        return PExp(self.N1[::-1], self.N0[::-1], self.t[-1] - self.t[::-1])

    @property
    def a(self):
        "eta(t) = a[i] exp(-(t[i + 1] - t)) b[i]) = 1 / (2 Ne(t))"
        return 1 / 2 / self.N1

    @property
    def b(self):
        "eta(t) = a[i] exp(-(t[i + 1]-t) b[i]) = 1 / (2 Ne(t))"
        # eta(t[i]) = a[i] exp(-b[i] dt[i]) = 1 / 2 / self.N0 =>
        return -jnp.log(1 / 2 / self.N0 / self.a) / jnp.diff(self.t)

    def __call__(self, u: jnp.ndarray):
        r"Evaluate eta(u)."
        t = self.t
        i = jnp.maximum(jnp.searchsorted(t, u) - 1, 0)  # t[j] <= u < t[j + 1]
        x = (t[i + 1] - u) / (t[i + 1] - t[i])
        return self.N1[i] * (self.N0[i] / self.N1[i]) ** x

    def R(self, u: jnp.ndarray):
        r"Evaluate R(u) = \int_0^u eta(s) ds"
        a = self.a
        b = self.b
        t = self.t
        dt = jnp.diff(jnp.minimum(t, u))
        ui = jnp.where(u < t[:-1], t[:-1], jnp.where(t[1:] < u, t[1:], u))
        integrals = a / b * jnp.exp(-b * (t[1:] - ui)) * -jnp.expm1(-b * dt)
        const = jnp.isclose(self.N0, self.N1)
        integrals = jnp.where(const, a * dt, integrals)
        return integrals.sum()

    def exp_integral(self, t0: float, t1: float, c: float = 1.0):
        r"""Compute the integral $\int_t0^t1 exp[-c * (R(t) - R(t0))] dt$ for $R(t) = \int_0^s eta(s) ds$.

        Args:
            c: The constant multiplier of R(t) in the integral.
        Returns:
            The value of the integral.
        """
        Rt0 = self.R(t0)

        def f(N0i, N1i, ti, ti1):
            # \int_ti^ti1 exp(-c R(t)) dt
            # = \int_ti^ti1 exp(-c R(ti) - c \int_ti^t eta(s) ds) dt
            # = \int_ti^ti1 exp(-c R(ti) - c \int_ti^t (1/2N0) ds) dt, if N0=N1
            # = exp(-c R(ti)) \int_ti^ti1 exp(-c (t - ti) (1/2N0) ds) dt
            # = exp(-c R(ti)) (N0/c) -expm1(-c / N0) dt)
            i1 = (
                jnp.exp(-c * (self.R(ti) - Rt0))
                * (2 * N0i / c)
                * -jnp.expm1(-c / (2 * N0i) * (ti1 - ti))
            )
            x1 = jnp.linspace(ti, ti1, 1000)
            x2 = jnp.linspace(x1[1], x1[-1], 1000)
            x = jnp.sort(jnp.concatenate([x1, x2]))
            i2 = jnp.trapezoid(jnp.exp(-c * (vmap(self.R)(x) - Rt0)), x)
            # ti1 might be +inf, but in that case we assume that N0i=N1i
            # (constant growth in last epoch)
            return jnp.where(jnp.isclose(N0i, N1i) | jnp.isinf(ti1), i1, i2)

        tm = self.t.clip(t0, t1)
        return vmap(f)(self.N0, self.N1, tm[:-1], tm[1:]).sum()
