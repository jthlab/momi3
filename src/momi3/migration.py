import itertools as it

import diffrax as dfx
import jax
import jax.numpy as jnp
import scipy.sparse as sps
from jax import jacfwd, vmap
from jax.experimental.sparse import BCOO

from .common import Axes, Ne_t, Population
from .kronprod import GroupedKronProd
from .momints import _drift, _migration, _mutation
from .spexpm import expmv


def lift_cm_aux(
    axes: Axes, migration_pairs: list[tuple[Population, Population]]
) -> dict:
    # compute the transition matrices for the dimensions involved in the lift. this function doesn't know about the
    # other dimensions that are not being lifted.
    tm = {}
    pops = {x for ab in migration_pairs for x in ab}
    tm["drift"] = {pop: _drift(axes[pop] - 1) for pop in pops}
    tm["mut"] = {pop: _mutation(axes[pop] - 1) for pop in pops}
    tm["mig"] = {
        (p1, p2): _migration((axes[p1] - 1, axes[p2] - 1)) for p1, p2 in migration_pairs
    }
    # convert sparse matrices from scipy to JAX format
    tm = jax.tree_util.tree_map(
        lambda A: BCOO.from_scipy_sparse(A) if isinstance(A, sps.spmatrix) else A, tm
    )
    tm["axes"] = axes
    return tm


def _e0_like(pl):
    return jnp.zeros(pl.size).at[0].set(1.0).reshape(pl.shape)


def lift_cm(params: dict, t: tuple[float, float], pl: jnp.ndarray, axes, aux):
    # Ne = params["Ne"]
    # const = all(not isinstance(Ne[pop], tuple) for pop in Ne)
    if False:  # const:
        f = _lift_cm_const
    else:
        f = _lift_cm_exp
    return f(params, t, pl, axes, aux)


def _A(s, y, args):
    theta, Q_mig, Q_mut, Q_drift, dims, axes, Ne, t, aux = args
    coal = {}
    for pop in Ne:
        i = list(axes).index(pop)
        if isinstance(Ne[pop], tuple):
            N0, N1 = Ne[pop]
            coal[i] = 1.0 / (4 * Ne_t(N0, N1, t[0], t[1], s))
        else:
            # Ne is a float, signifying constant Ne
            coal[i] = 1.0 / (4 * Ne[pop])
    # multiply each entry of the drift tensor by the coalescent rate
    new_A = []
    for Ai in Q_drift.A:
        assert len(Ai) == 1
        ((i, Aij),) = Ai.items()
        new_A.append({i: coal[i] * Aij})
    Qd = Q_drift._replace(A=new_A)
    Q = Q_mig + Qd + Q_mut * theta
    return Q @ y


def _lift_cm_exp(params, t, pl, axes, aux):
    # population sizes are changing, so we have to use a differential
    # equation solver
    dims = pl.shape
    Ne = params["Ne"]
    Q_drift = _Q_drift(dims, axes, {p: 1.0 for p in axes}, aux)
    Q_mig, Q_mut = _Q_mig_mut(dims, axes, params["mig"], aux, tr=False)
    # WORKAROUND: calling Q_mig.T below gives me an error, impossibly deep stack trace having to do with diffrax,
    # bcoo_sparse, vjp, etc. etc. "manually" transposing before multiplying with any traced migration params seems
    # to be the key.
    Q_mig_T, _ = _Q_mig_mut(dims, axes, params["mig"], aux, tr=True)
    term = dfx.ODETerm(_A)
    # solver = dfx.Kvaerno4(
    #     nonlinear_solver=dfx.NewtonNonlinearSolver(rtol=1e-5, atol=1e-6),
    #     scan_stages=True
    # )
    solver = dfx.Tsit5(scan_stages=True)
    ssc = dfx.PIDController(rtol=1e-6, atol=1e-7)

    def solve(theta, y0, args):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[1],
            dt0=(t[1] - t[0]) / 50.0,
            y0=y0,
            args=(theta,) + args,
            stepsize_controller=ssc,
        ).ys[0]

    primal_args = (Q_mig_T, Q_mut.T, Q_drift.T, dims, axes, Ne, t, aux)
    plp = solve(0.0, pl, primal_args)
    tangent_args = (Q_mig, Q_mut, Q_drift, dims, axes, Ne, t, aux)
    e0 = _e0_like(pl)
    # etbl = jax.jacrev(solve)(0.0, e0, tangent_args)
    # jacrev(solve) does not work inside grad due to limitations with diffrax. so we settle for finite differences.
    eps = 1e-5
    res = vmap(solve, (0, None, None))(jnp.array([eps, -eps]), e0, tangent_args)
    etbl = (res[0] - res[1]) / (2 * eps)
    return plp.clip(0.0, 1.0), etbl.clip(0.0)


def _lift_cm_const(params: dict, t: tuple[float, float], pl: jnp.ndarray, axes, aux):
    """
    Lift partial likelihoods under continuous migration.

    Args:
        params: dict of parameters for migration model, one per population
        t: length of time to lift
        pl: partial likelihoods
        axes: dict mapping populations to be lifted to their positions in pl
        aux: the output of lift_cm_aux (see below)

    Returns:
        Tuple (lifted_likelihood, phi) where phi contains the expected branch lengths subtending each entry of the JSFS
        for these populations.
    """
    dt = t[1] - t[0]
    dims = pl.shape
    Ne = params["Ne"]
    coal = {pop: 1.0 / Ne[pop] for pop in Ne}
    Q_drift = _Q_drift(
        dims,
        axes,
        coal,
        aux,
    )
    Q_mig, Q_mut = _Q_mig_mut(
        dims,
        axes,
        params["mig"],
        aux,
    )
    Q_lift = (Q_mig + Q_drift) * dt
    pl_lift = expmv(Q_lift.T, pl)
    assert pl_lift.shape == pl.shape
    # now compute the expected branch lengths
    e0 = _e0_like(pl)

    def f(theta):
        # note: Q_mut * (...) has to be implemented as multiplication from the right order for it to work with traced
        # jax code
        Q = Q_lift + Q_mut * theta * dt
        v = expmv(Q, e0)
        return v

    etbl = jacfwd(f)(0.0).at[(0,) * pl.ndim].set(0.0)
    return pl_lift, etbl


def _Q_drift(
    dims, axes, coal_rates, aux
) -> tuple[GroupedKronProd, GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    s = list(coal_rates)  # these are the populations participating in the migration
    i = list(axes).index
    return GroupedKronProd(
        [{i(ss): aux["drift"][ss] * coal_rates[ss]} for ss in s], dims
    )


def _Q_mig_mut(
    dims, axes, mig_mat, aux, tr=False
) -> tuple[GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    s = list(aux["mut"])
    i = list(axes).index
    Q_mut = GroupedKronProd([{i(ss): aux["mut"][ss]} for ss in s], dims)
    # migration matrix is a bit trickier
    terms = []
    for s1, s2 in it.product(s, repeat=2):
        if (s1, s2) in mig_mat:
            m_ij = mig_mat[s1, s2]
            u1, u2 = aux["mig"][s1, s2]
            if tr:
                u1, u2 = [{k: v.T for k, v in x.items()} for x in (u1, u2)]
            i1, i2 = map(list(axes).index, (s1, s2))
            terms.append({i1: m_ij * u1[0], i2: u1[1]})
            terms.append({i2: m_ij * u2[1]})
    Q_mig = GroupedKronProd(terms, dims)
    return Q_mig, Q_mut
