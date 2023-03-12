# Sparse matrix exponential
import operator
from functools import reduce, singledispatch

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_dot_general_p, sparsify

from momi3.kronprod import KronProd, tr


def _save_sparse_matmul(prim, *_, **__) -> bool:
    # rematerialization policy to save sparse matrix multiply results only
    return prim is bcoo_dot_general_p


@singledispatch
def eye_like(A: BCOO):
    return jax.experimental.sparse.eye(A.shape[0])


@eye_like.register
def _(A: jnp.ndarray):
    return jnp.eye(A.shape[0])


@eye_like.register
def _(A: np.ndarray):
    return np.eye(A.shape[0])


@singledispatch
def expmv(A: BCOO, B: jnp.ndarray) -> jnp.ndarray:
    "(sparse) matrix exponential times vector"
    n = A.shape[0]
    I_n = eye_like(A)
    mu = tr(A) / n
    A = A - mu * I_n
    A_1norm = sparsify(abs)(A).sum(1).todense().max()
    return _expmv_inner(A, 1.0, B, A_1norm, mu)


@expmv.register
def _(A: KronProd, B: jnp.ndarray) -> jnp.ndarray:
    A = A._replace(
        dims=B.shape
    )  # FIXME: this is a hack to prevent traced values from showing up in dims.
    n = reduce(operator.mul, A.dims)
    mu = tr(A) / n
    # subtract I*mu from tensor product
    (i, Ai), *_ = A.A[0].items()
    A = A._replace(A=A.A + [{i: -mu * eye_like(Ai)}])
    A_1norm = A.bound_norm1()
    return _expmv_inner(A, 1.0, B, A_1norm, mu)


@expmv.register
def _(A: np.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    n = A.shape[0]
    I_n = np.eye(n)
    mu = A.trace() / n
    A = A - mu * I_n
    A_1norm = np.linalg.norm(A, 1)
    return _expmv_inner(A, 1.0, B, A_1norm, mu)


def _expmv_inner(A, t, B, A_1norm, mu):
    if A_1norm.dtype == jnp.float32:
        theta = THETA_SINGLE
    else:
        assert A_1norm.dtype == jnp.float64
        theta = THETA_DOUBLE
    m_max = len(theta)
    m = jnp.arange(1, m_max + 1)
    u = jnp.ceil(A_1norm / theta).astype(int)
    m_star = (m * u).argmin()
    s = u[m_star]

    # given m_star, s_star the algorithm is really simple
    tol = jnp.finfo(A_1norm.dtype).eps

    def _infnorm(B):
        return jnp.linalg.norm(B.reshape(-1), jnp.inf)

    eta = jnp.exp(t * mu / s)

    def f1(i, accum):
        def g1(j, tup):
            c1, F, B, br = tup
            coeff = t / (s * (j + 1))
            B1 = coeff * (A @ B)
            B = B1.astype(B.dtype)
            c2 = _infnorm(B)
            F += B
            br = c1 + c2 <= tol * _infnorm(F)
            return c2, F, B, br

        def g2(j, tup):
            return tup

        def g(j, tup):
            c1, F, B, br = tup
            return lax.cond((j < m_star) & (~br), g1, g2, j, tup)

        F, B = accum
        c1 = _infnorm(B)
        _, F, _, _ = lax.fori_loop(0, m_max, g, (c1, F, B, False))
        F *= eta
        B = F
        return (F, B)

    def f2(i, accum):
        return accum

    # @partial(checkpoint, policy=_save_sparse_matmul)
    def f(i, accum):
        return lax.cond(i < s, f1, f2, i, accum)

    F, _ = lax.fori_loop(0, len(theta), f, (B, B))

    return F


THETA_DOUBLE = jnp.array(
    [
        2.22044605e-16,
        2.58095680e-08,
        1.38634787e-05,
        3.39716884e-04,
        2.40087636e-03,
        9.06565641e-03,
        2.38445553e-02,
        4.99122887e-02,
        8.95776020e-02,
        1.44182976e-01,
        2.14235807e-01,
        2.99615891e-01,
        3.99777534e-01,
        5.13914694e-01,
        6.41083523e-01,
        7.80287426e-01,
        9.30532846e-01,
        1.09086372e00,
        1.26038106e00,
        1.43825260e00,
        1.62371595e00,
        1.81607782e00,
        2.01471078e00,
        2.21904887e00,
        2.42858252e00,
        2.64285346e00,
        2.86144963e00,
        3.08400054e00,
        3.31017284e00,
        3.53966635e00,
        3.77221050e00,
        4.00756109e00,
        4.24549744e00,
        4.48581986e00,
        4.72834735e00,
        4.97291563e00,
        5.21937537e00,
        5.46759063e00,
        5.71743745e00,
        5.96880263e00,
        6.22158266e00,
        6.47568274e00,
        6.73101590e00,
        6.98750228e00,
        7.24506843e00,
        7.50364669e00,
        7.76317466e00,
        8.02359473e00,
        8.28485363e00,
        8.54690205e00,
        8.80969427e00,
        9.07318789e00,
        9.33734351e00,
        9.60212447e00,
        9.86749668e00,
        1.01334283e01,
        1.03998897e01,
        1.06668532e01,
        1.09342929e01,
        1.12021845e01,
        1.14705053e01,
        1.17392341e01,
        1.20083509e01,
        1.22778370e01,
        1.25476748e01,
        1.28178476e01,
        1.30883399e01,
        1.33591369e01,
        1.36302250e01,
        1.39015909e01,
        1.41732223e01,
        1.44451076e01,
        1.47172357e01,
        1.49895963e01,
        1.52621795e01,
        1.55349758e01,
        1.58079765e01,
        1.60811732e01,
        1.63545578e01,
        1.66281227e01,
        1.69018609e01,
        1.71757655e01,
        1.74498298e01,
        1.77240478e01,
        1.79984136e01,
        1.82729215e01,
        1.85475662e01,
        1.88223426e01,
        1.90972458e01,
        1.93722711e01,
        1.96474142e01,
        1.99226707e01,
        2.01980367e01,
        2.04735082e01,
        2.07490816e01,
        2.10247533e01,
        2.13005199e01,
        2.15763782e01,
        2.18523250e01,
        2.21283574e01,
    ]
)

THETA_SINGLE = jnp.array(
    [
        1.19209280e-07,
        5.97885889e-04,
        1.12338647e-02,
        5.11661936e-02,
        1.30848716e-01,
        2.49528932e-01,
        4.01458242e-01,
        5.80052463e-01,
        7.79511337e-01,
        9.95184079e-01,
        1.22347954e00,
        1.46166151e00,
        1.70764853e00,
        1.95985059e00,
        2.21704439e00,
        2.47828088e00,
        2.74281711e00,
        3.01006636e00,
        3.27956121e00,
        3.55092621e00,
        3.82385743e00,
        4.09810697e00,
        4.37347131e00,
        4.64978222e00,
        4.92689984e00,
        5.20470723e00,
        5.48310609e00,
        5.76201341e00,
        6.04135876e00,
        6.32108213e00,
        6.60113218e00,
        6.88146485e00,
        7.16204215e00,
        7.44283129e00,
        7.72380381e00,
        8.00493499e00,
        8.28620327e00,
        8.56758983e00,
        8.84907819e00,
        9.13065388e00,
        9.41230420e00,
        9.69401796e00,
        9.97578527e00,
        1.02575974e01,
        1.05394467e01,
        1.08213263e01,
        1.11032302e01,
        1.13851530e01,
        1.16670899e01,
        1.19490366e01,
        1.22309896e01,
        1.25129453e01,
        1.27949008e01,
        1.30768536e01,
        1.33588011e01,
        1.36407415e01,
        1.39226727e01,
        1.42045932e01,
        1.44865015e01,
        1.47683963e01,
    ]
)
