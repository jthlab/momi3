import numpy as np
from moments import Integration, Jackknife
from pytest import fixture

from momi3.common import Axes
from momi3.lemmas.lift import _aux_single, _lift1
from momi3.migration import (
    _lift_cm_const,
    _lift_cm_exp,
    _Q_drift,
    _Q_mig_mut,
    lift_cm_aux,
)


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
    mig_mat = {pops[0]: 0.01}
    Qmig1, _ = _Q_mig_mut(dims, axes, mig_mat, aux)
    M0, M1 = [np.asarray(x.todense()) for x in (Qmig0, Qmig1.materialize())]
    np.testing.assert_allclose(M0, M1, atol=1e-6)


def test_Qdrift(axes, n, dims, rng):
    Npop = np.ones(2)
    vd = Integration._calcD(dims)
    (D0,) = Integration._buildD(vd, dims, Npop)
    pops = [("A", "B")]
    aux = lift_cm_aux(axes, pops)
    coal_rates = {s: 1 / 4.0 for s in pops[0]}
    D1 = _Q_drift(
        dims, axes, coal_rates, aux
    )  # moments scales by 4 * Ne but we scale be Ne
    M0, M1 = [np.asarray(x.todense()) for x in (D0, D1.materialize())]
    np.testing.assert_allclose(M0, M1, atol=1e-6)


def test_lift_eq_const_exp0(rng):
    "test that lifting a 2-tensor with constant population size is the same as lifting with exponential growth(g=0)"
    pl = rng.uniform(size=(8, 5))  # 5 is the minimum size for the migration matrices
    params = {"Ne": {"A": (1.0, 1.0), "B": (2.0, 2.0)}, "mig": {("A", "B"): 0.01}}
    axes = Axes(zip("AB", pl.shape))
    aux = lift_cm_aux(axes, [("A", "B")])
    t = (0.0, 1.0)
    pl1 = _lift_cm_exp(params, t, pl, axes, aux)
    # params has different form for const
    params["Ne"] = dict(A=1.0, B=2.0)
    pl2 = _lift_cm_const(params, t, pl, axes, aux)
    np.testing.assert_allclose(pl1, pl2, atol=1e-6, rtol=1e-6)


def test_lift_eq_exp_m0(rng):
    "test that jointly lifting a 2-tensor with exponential population size is the same as lifting with migration=0"
    n_A = 7
    n_B = 4
    t = [0.0, 21.2e3]
    pl = rng.uniform(
        size=(n_A + 1, n_B + 1)
    )  # 5 is the minimum size for the migration matrices
    axes = Axes(zip("AB", pl.shape))
    aux = lift_cm_aux(axes, [("A", "B")])
    params = {"Ne": {"A": 12300, "B": (54090.0, 510.0)}, "mig": {("A", "B"): 0.0}}
    plp_mig, etbl_mig = _lift_cm_exp(params, t, pl, axes, aux)

    plp_nomig = pl
    etbl_nomig = np.zeros_like(pl)
    for pop in axes:
        nv = axes[pop] - 1
        aux = _aux_single(nv)
        i = list(axes).index(pop)
        plp_nomig, e = _lift1(
            plp_nomig, i, params["Ne"][pop], t[1] - t[0], **aux, terminal=False
        )
        sl = [0] * 2
        sl[i] = slice(None)
        etbl_nomig[tuple(sl)] += e

    np.testing.assert_allclose(plp_mig, plp_nomig, rtol=1e-5)
    np.testing.assert_allclose(etbl_mig, etbl_nomig, rtol=1e-5)
