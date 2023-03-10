import LinearSystem_2D
import moments
import numpy as np
import scipy.sparse as sps
from moments.LinearSystem import calcD
from pytest import fixture

from momi3.momints import (
    _downsample,
    _drift,
    _ibis,
    _jackknife0,
    _migration,
    _mutation,
    _Qm_matrix,
)


def assert_sp_eq(X, Y):
    assert (abs(X - Y) > 1e-6).sum() == 0


@fixture(params=[5, 10, 25, 50])
def n(request):
    return request.param


def test_Qm(n, rng):
    dims = 1 + np.array([n, n + 1, n + 2])
    M = rng.random(size=(3, 3))
    ljk = [moments.Jackknife.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
    vm = moments.Integration._calcM(dims, ljk)
    Q1 = moments.Integration._buildM(vm, dims, M)
    Q2 = _Qm_matrix(dims, M.T)
    assert_sp_eq(Q1[0], Q2[(0, 1)])
    assert_sp_eq(Q1[1], Q2[(0, 2)])
    assert_sp_eq(Q1[2], Q2[(1, 2)])


def test_drift_operator(n):
    (D_moments,) = calcD([n + 1])
    D_jax = _drift(n)
    assert_sp_eq(D_moments, D_jax)


def test_drift_operator_multi():
    ns = [5, 10]
    D_moments = calcD(ns)
    D_jax = [_drift(n - 1) for n in ns]
    ident = [sps.eye(A.shape[0]) for A in D_jax]
    assert_sp_eq(D_moments[0], sps.kron(D_jax[0], ident[1]))
    assert_sp_eq(D_moments[1], sps.kron(ident[0], D_jax[1]))


def test_ibis(n):
    ibis1 = np.array([moments.Jackknife.index_bis(i + 1, n) for i in range(n)])
    ibis2 = _ibis(n)
    i = np.arange(1, n + 1)
    err1 = ibis1 / n - i / (n + 1)
    err2 = ibis2 / n - i / (n + 1)
    assert np.all(err2 <= err1)
    assert np.all(ibis1 == ibis2)


def test_mutation(n):
    if n > 30:
        return
    dims = np.array([n + 1, n + 2])
    m1 = np.asarray(LinearSystem_2D.calcB_FB1(dims, 1, 1).todense())
    m2 = np.asarray(LinearSystem_2D.calcB_FB2(dims, 1, 1).todense())
    M1 = _mutation(n).todense()
    np.testing.assert_allclose(M1, m1.reshape(n + 1, n + 2, n + 1, n + 2)[:, 0, :, 0])
    M2 = _mutation(n + 1).todense()
    np.testing.assert_allclose(M2, m2.reshape(n + 1, n + 2, n + 1, n + 2)[0, :, 0, :])


def test_jacknife(n):
    J_moments = moments.Jackknife.calcJK13(n)
    J_jax = _jackknife0(n)
    np.testing.assert_allclose(J_moments, J_jax.todense())


def test_calc_M(n):
    d = d1, d2 = np.array([n, n + 1])
    ljk = _jackknife0(d1 - 1).todense()
    M1 = moments.LinearSystem_2D.calcM_2(d, ljk)
    n1, n2 = d - 1
    tps = _migration([n1, n2])
    M2 = sum(sps.kron(r.get(0, sps.eye(d1)), r.get(1, sps.eye(d2))) for r in tps)
    assert_sp_eq(M1, M2)
    ljk = _jackknife0(d2 - 1).todense()
    M1 = moments.LinearSystem_2D.calcM_1(d, ljk)
    tps = _migration([n2, n1])
    mats = [sps.kron(r.get(1, sps.eye(d1)), r.get(0, sps.eye(d2))) for r in tps]
    M2 = sum(mats)
    assert_sp_eq(M1, M2)


def test_downsample(n):
    fss = []
    for i in [0, 1]:
        ns = [n, n + i]
        sts = moments.LinearSystem_1D.steady_state_1D(sum(ns))
        fs = moments.Spectrum(sts)
        fss.append(moments.Manips.split_1D_to_2D(fs, ns[0], ns[1]))
    _downsample(n + 1)
