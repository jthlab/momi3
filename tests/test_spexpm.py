import jax
import numpy as np
from jax.experimental.sparse import BCOO
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from momi3.kronprod import KronProd
from momi3.spexpm import expmv


def test_spexpm(rng):
    A, B = rng.random(size=(2, 10, 10))
    Asp = BCOO.fromdense(A)
    P1 = expmv(Asp, B)
    P2 = expm_multiply(A, B)
    np.testing.assert_allclose(P1, P2, rtol=1e-6)


def test_spexpm_kron(rng):
    M = rng.random(size=(3, 4, 4))
    dims = (4,) * len(M)
    B = rng.random(size=dims)
    K1 = KronProd([{i: m} for i, m in enumerate(M)], dims)
    K2 = KronProd([{i: m for i, m in enumerate(M)}], dims)
    for K in K1, K2:
        P1 = expmv(K, B)
        P2 = expmv(K.materialize(), B.reshape(-1)).reshape(dims)
        np.testing.assert_allclose(P1, P2, rtol=1e-6)


# flake8: noqa
def test_spexpm_numerical_bug1():
    # fmt: off
    data = np.array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,
        0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1, -0.2, -0.2,
       -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0. ,
       -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ,
       -0. , -0. , -0. , -0. , -0. , -0. ,  0.3,  0.3,  0.3,  0.3,  0.4,
        0.4,  0.4,  0.4,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,
        0.4,  0.4,  0.4,  0.4,  0.3,  0.3,  0.3,  0.3, -0.6, -0.8, -0.6,
       -0.6, -0.8, -0.6, -0.6, -0.8, -0.6, -0.6, -0.8, -0.6,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
        0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. , -0. ,
       -0. , -0. , -0. , -0. , -0. , -0. , -0. ])
    i, j = indicesT = np.array([
        [ 5, 10,  6, 11,  7, 12,  8, 13,  9, 14,  5, 10,  6, 11,  7,
              12,  8, 13,  9, 14,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
               5, 10, 15,  6, 11, 16,  7, 12, 17,  8, 13, 18,  1,  6, 11,
               2,  7, 12,  3,  8, 13,  4,  9, 14,  1,  2,  3,  4,  5,  6,
               7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  1,  6, 11,
              16,  2,  7, 12, 17,  3,  8, 13, 18,  1,  6, 11, 16,  2,  7,
              12, 17,  3,  8, 13, 18,  1,  2,  3,  6,  7,  8, 11, 12, 13,
              16, 17, 18,  1,  6, 11,  2,  7, 12,  3,  8, 13,  4,  9, 14,
               5, 10, 15,  6, 11, 16,  7, 12, 17,  8, 13, 18,  1,  2,  3,
               4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
             [ 0,  5,  1,  6,  2,  7,  3,  8,  4,  9, 10, 15, 11, 16, 12,
              17, 13, 18, 14, 19,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
               6, 11, 16,  7, 12, 17,  8, 13, 18,  9, 14, 19,  0,  5, 10,
               1,  6, 11,  2,  7, 12,  3,  8, 13,  1,  2,  3,  4,  5,  6,
               7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0,  5, 10,
              15,  1,  6, 11, 16,  2,  7, 12, 17,  2,  7, 12, 17,  3,  8,
              13, 18,  4,  9, 14, 19,  1,  2,  3,  6,  7,  8, 11, 12, 13,
              16, 17, 18,  6, 11, 16,  7, 12, 17,  8, 13, 18,  9, 14, 19,
               0,  5, 10,  1,  6, 11,  2,  7, 12,  3,  8, 13,  1,  2,  3,
               4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]],            dtype=int)
    # fmt: on
    A_jax = BCOO((data, indicesT.T), shape=(20, 20))
    A_scipy = coo_matrix((data, (i, j)), (20, 20))
    t = 10
    B = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )
    P1 = expmv(A_jax * t, B.T)
    P2 = expm_multiply(A_scipy * t, B.T)
    np.testing.assert_allclose(P1, P2, rtol=1e-5)


def test_spexpm_eq_t(rng):
    X = rng.random(size=(10, 10))
    X -= np.diag(np.diag(X))
    X -= np.diag(X.sum(1))
    v = rng.random(size=10)
    v /= v.sum()
    t = 1000.0
    p0 = expm_multiply(X.T * t, v)
    # p1 = expmv(X.T, t, v)
    with jax.experimental.enable_x64(True):
        p2 = expmv(X.T * t, v)
    np.testing.assert_allclose(p0, p2)
