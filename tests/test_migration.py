import numpy as np
from moments import Integration, Jackknife
from pytest import fixture

from momi3.common import Axes
from momi3.migration import _Q_matrix, lift_cm_aux


@fixture
def axes():
    return Axes({"A": 5, "B": 6})


@fixture
def n():
    return 5


@fixture
def dims(axes):
    return [a for a in axes.values()]


def test_Qmig(axes, n, dims, rng):
    M = np.zeros([2, 2])
    M[0, 1] = 0.01
    ljk = [Jackknife.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
    vm = Integration._calcM(dims, ljk)
    (Qmig0,) = Integration._buildM(vm, dims, M.T)
    Qmig0 = Qmig0
    pops = [("A", "B")]
    aux = lift_cm_aux(axes, pops)
    coal_rates = {s: 1 / 2.0 for s in pops[0]}
    Qmig1, _, _ = _Q_matrix(dims, pops[0], coal_rates, {pops[0]: 0.01}, aux)
    M0, M1 = [np.asarray(x.todense()) for x in (Qmig0, Qmig1.materialize())]
    np.testing.assert_allclose(M0, M1, atol=1e-6)


def test_Qdrift(axes, n, dims, rng):
    Npop = np.ones(2)
    vd = Integration._calcD(dims)
    (D0,) = Integration._buildD(vd, dims, Npop)
    pops = [("A", "B")]
    aux = lift_cm_aux(axes, pops)
    coal_rates = {s: 1 / 4.0 for s in pops[0]}
    _, D1, _ = _Q_matrix(
        dims, pops[0], coal_rates, {}, aux
    )  # moments scales by 4 * Ne but we scale be Ne
    M0, M1 = [np.asarray(x.todense()) for x in (D0, D1.materialize())]
    np.testing.assert_allclose(M0, M1, atol=1e-6)
