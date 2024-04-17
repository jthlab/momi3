from functools import reduce

import numpy as np
import scipy.sparse as sps
from jax.experimental.sparse import BCOO
from pytest import fixture

from momi3.kronprod import GroupedKronProd, KronProd


def assert_sp_eq(A, B):
    np.testing.assert_allclose(A.todense(), B.todense(), atol=1e-6)


@fixture
def n():
    return 5


@fixture
def K_kron(n, rng):
    M = rng.random(size=(3, n, n))
    M = BCOO.fromdense(M)
    dims = (n,) * 3
    return KronProd([{i: m for i, m in enumerate(M)}], dims)


@fixture
def K_kronsum(n, rng):
    M = rng.random(size=(3, n, n))
    M = BCOO.fromdense(M)
    dims = (n,) * 3
    return KronProd([{i: m} for i, m in enumerate(M)], dims)


def test_transpose(K_kron, K_kronsum):
    for K in K_kron, K_kronsum:
        assert_sp_eq(K.T.materialize(), K.materialize().T)


def test_materialize(K_kron, K_kronsum):
    M = [K_kron.A[0][i].todense() for i in range(len(K_kron.A[0]))]
    np.testing.assert_allclose(
        K_kron.materialize().todense(), reduce(np.kron, M), atol=1e-6
    )

    M = [K_kronsum.A[i][i].todense() for i in range(len(K_kronsum.A))]
    np.testing.assert_allclose(
        K_kronsum.materialize().todense(),
        # sps.kronsum is backwards from the usual definition
        reduce(sps.kronsum, M[::-1]).todense(),
        atol=1e-6,
    )


def test_trace(K_kron, K_kronsum):
    for K in K_kron, K_kronsum:
        np.testing.assert_allclose(
            K.materialize().todense().trace(),
            K.trace(),
            atol=1e-4,
        )


def test_matmul(rng: np.random.Generator, n):
    M = BCOO.fromdense(rng.random(size=(3, n, n)))
    dims = (n,) * 3
    N = rng.random(size=dims)
    K = KronProd([{i: m for i, m in enumerate(M)}], dims)
    np.testing.assert_allclose(
        K @ N, (K.materialize() @ N.reshape(-1)).reshape(dims), rtol=1e-6
    )

    K = KronProd([{i: m} for i, m in enumerate(M)], dims)
    np.testing.assert_allclose(
        K @ N, (K.materialize() @ N.reshape(-1)).reshape(dims), rtol=1e-6
    )


def test_group2(rng):
    A, B = map(BCOO.fromdense, [rng.random(size=(n, n)) for n in (3, 4)])
    K = KronProd([{0: A, 1: B}, {0: A}, {1: B}], (3, 4))
    X = rng.random(size=(3, 4))
    np.testing.assert_allclose(K @ X, GroupedKronProd(K.A, K.dims) @ X, rtol=1e-6)


def test_I2(rng):
    A, B = map(BCOO.fromdense, [np.eye(n) for n in (3, 4)])
    K = KronProd([{0: A, 1: B}, {0: A}, {1: B}], (3, 4))
    X = rng.random(size=(3, 4))
    np.testing.assert_allclose(K @ X, 3 * X)


def test_group3(rng):
    A, B, C = map(BCOO.fromdense, [rng.random(size=(n, n)) for n in (3, 4, 5)])
    K = KronProd(
        [{0: A}, {1: B}, {2: C}, {0: A, 1: B}, {0: A, 2: C}, {1: B, 2: C}], (3, 4, 5)
    )
    X = rng.random(size=(3, 4, 5))
    np.testing.assert_allclose(K @ X, GroupedKronProd(K.A, K.dims) @ X, rtol=1e-6)
