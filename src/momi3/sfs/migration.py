import diffrax as dfx
import jax
import jax.numpy as jnp
import itertools as it
import scipy.sparse as sps
from typing import NamedTuple, Any
from jax import jacfwd
from jax.experimental.sparse import BCOO
from scipy.sparse.linalg._expm_multiply import _expm_multiply_simple

from ..common import Axes, Population
from .kronprod import GroupedKronProd
from .momints import _drift, _migration, _mutation

jax.config.update("jax_bcoo_cusparse_lowering", True)


class MigrationMatrix(NamedTuple):
    params: dict[str, Any]
    axes: Axes

    @property
    def jump_ts(self):
        return jnp.concatenate(
            [
                jnp.array([m["start_time"], m["end_time"]])
                for m in self.params["migrations"]
            ]
        )

    def __call__(self, t: float):
        M_ij = {}
        for p1, p2 in it.product(self.axes, repeat=2):
            ms = [
                m
                for m in self.params["migrations"]
                if m["source"] == p1 and m["dest"] == p2
            ]

            if not ms:
                continue

            A = jnp.array([[m["rate"], m["end_time"], m["start_time"]] for m in ms])
            i = A[:, 1].argsort()
            r, t_start, t_end = A[i].T

            def f(t, t_start=t_start, t_end=t_end, r=r):
                j = jnp.searchsorted(t_start, t, side="right") - 1
                return jnp.where((t_start[j] <= t) & (t < t_end[j]), r[j], 0.0)

            M_ij[p1, p2] = f

        a = len(self.axes)
        M = [[0.0] * a for _ in range(a)]
        for (i1, p1), (i2, p2) in it.product(enumerate(self.axes), repeat=2):
            if (p1, p2) in M_ij:
                M[i2][i1] = M_ij[p1, p2](t)
            else:
                M[i2][i1] = 0.0
        # rate of entering coalescent state
        # rate of coalescing equals assign
        M = jnp.array(M)
        M -= jnp.diag(M.sum(1))
        return M


def _dense_expmv(A, v, t):
    result_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
    return jax.pure_callback(_expm_multiply_simple, result_shape, A, v, t)


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
    def f(A):
        return A.todense()
        if isinstance(A, sps.spmatrix):
            ret = BCOO.from_scipy_sparse(A).sort_indices()
            ret.unique_indices = True
            return ret
        return A

    tm = jax.tree.map(f, tm)
    tm["axes"] = axes
    return tm


def _e0_like(pl):
    return jnp.zeros(pl.size).at[0].set(1.0).reshape(pl.shape)


def lift_cm(params: dict, t: tuple[float, float], pl: jnp.ndarray, axes, aux):
    # Ne = params["Ne"]
    # all(not isinstance(Ne[pop], tuple) for pop in Ne)
    # if False:
    #     logger.debug("using sparse matrix exponentiation for {}", aux)
    #     f = _lift_cm_const
    # else:
    #     logger.debug("using diffeq solver for {}", aux)
    #     f = _lift_cm_exp
    return _lift_cm_exp(params, t, pl, axes, aux)


def _A(s, y, args):
    f_Q_mig, Q_mut, Q_drift, dims, axes, aux, etas = args
    Q_mig = f_Q_mig(s)
    coal = {}
    for pop in axes:
        i = list(axes).index(pop)
        coal[i] = 1 / (4 * etas[pop](s))
        # if isinstance(Ne[pop], tuple):
        #     N0, N1 = Ne[pop]
        #     coal[i] = 1.0 / (4 * Ne_t(N0, N1, t[0], t[1], s))
        # else:
        #     # Ne is a float, signifying constant Ne
        #     coal[i] = 1.0 / (4 * Ne[pop])
    # multiply each entry of the drift tensor by the coalescent rate
    new_A = []
    for Ai in Q_drift.A:
        assert len(Ai) == 1
        ((i, Aij),) = Ai.items()
        new_A.append({i: coal[i] * Aij})
    Qd = Q_drift._replace(A=new_A)
    Q0 = Q_mig + Qd
    if isinstance(y, tuple):
        return Q0 @ y[0] + Q_mut @ y[1], Q0 @ y[1]
    return Q0 @ y


def _lift_cm_exp(params, t, pl, axes, aux):
    # population sizes are changing, so we have to use a differential
    # equation solver
    dims = pl.shape
    etas = params["etas"]
    Q_drift = _Q_drift(dims, axes, {p: 1.0 for p in etas}, aux)
    f_Q_mig, Q_mut = _Q_mig_mut(t[0], t[1], dims, axes, params["mig"], aux, tr=False)
    f_Q_mig_T, _ = _Q_mig_mut(t[0], t[1], dims, axes, params["mig"], aux, tr=True)

    solver = dfx.Tsit5()
    term = dfx.ODETerm(_A)

    def solve(y0, args):
        f_Q_mig = args[0]
        etas = args[-1]
        jump_ts = jnp.array([eta.t for eta in etas.values()])
        jump_ts = jnp.append(jump_ts, f_Q_mig.t)
        jump_ts = jnp.sort(jump_ts)
        ssc = dfx.PIDController(jump_ts=jump_ts, rtol=1e-5, atol=1e-5)
        res = dfx.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[1],
            dt0=(t[1] - t[0]) / 100,
            # dt0=None,
            y0=y0,
            args=args,
            stepsize_controller=ssc,
            max_steps=4096,
            # max_steps=16384,
            adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=10),
            # adjoint=dfx.BacksolveAdjoint(),
            # adjoint=dfx.DirectAdjoint(),
        )
        # jax.debug.print("number of steps: {}", res.stats["num_steps"])
        return res.ys

    primal_args = (f_Q_mig_T, Q_mut.T, Q_drift.T, dims, axes, aux, etas)
    plp = solve(pl, primal_args)[0]

    # compute d/dtheta x(t,theta)|{theta=0} using the forward sensitivity method.
    # we have x'(t, theta) = Q(t, theta) @ x(t, theta) and therefore
    # d/dtheta x'(t, theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    # = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # d/dt dx(t,theta)/dtheta d/dtheta x'(t,theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    #   = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # dF/dtheta = d(Q @ x)/dtheta = (Q_mut @ x)
    # the initial condition is d/dtheta(x(0, theta)) = 0.; x(0,theta) = e0

    # for computing branch length, we only need to track the populations that are involved in the migration
    involved = list(etas.keys())
    sh = tuple([pl.shape[i] if pop in involved else 1 for i, pop in enumerate(axes)])
    z = jnp.zeros(sh)
    e0 = z.at[(0,) * z.ndim].set(1.0)
    new_etas = {}
    for k, v in etas.items():
        new_etas[k] = lambda x: v(t[0] + t[1] - x)
        new_etas[k].t = t[0] + t[1] - v.t
    tangent_args = (f_Q_mig,)
    tangent_args += tuple([X._replace(dims=sh) for X in (Q_mut, Q_drift)])
    tangent_args += (
        sh,
        axes,
        aux,
        new_etas,
    )
    # time runs backwards here!
    res = solve((z, e0), tangent_args)
    etbl = res[0][0]
    inds = tuple([slice(None) if pop in involved else 0 for pop in axes])
    etbl = etbl[inds]
    for x in (0, -1):
        etbl = etbl.at[(x,) * pl.ndim].set(0.0)
    return plp, etbl


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
    coal = {pop: 1.0 / (4 * Ne[pop]) for pop in Ne}
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
    if False:

        def expmv(A, x):
            return (
                jax.scipy.linalg.expm(A, max_squarings=128) @ x.reshape(-1)
            ).reshape(x.shape)

        A = Q_lift.T.materialize().todense()
        pl_lift = expmv(A, pl)
    else:
        A = (Q_mig + Q_drift).T.materialize()
        plf = pl.reshape(-1)
        assert plf.shape == (A.shape[0],)
        pl_lift = _dense_expmv(A, plf, dt).reshape(pl)

    assert pl_lift.shape == pl.shape
    # now compute the expected branch lengths
    e0 = _e0_like(pl)

    def f(theta):
        # note: Q_mut * (...) has to be performed as right multiplication for it to work with traced
        # jax code
        Q = Q_lift + Q_mut * theta * dt
        A = Q.materialize().todense()
        return expmv(A, e0)

    etbl0 = jacfwd(f)(0.0)
    # d/dt expm(Q1 + t Q2) v |{t=0} ~= (Q1 + Q1 Q2) v

    # def f(Qsp, x):
    #     return (Qsp.materialize().todense() @ x.reshape(-1)).reshape(x.shape)

    # etbl0 = Q_lift @ e0 + Q_lift @ (Q_mut @ (dt * e0))
    # etbl0 = f(Q_lift, e0) + f(Q_lift, f(Q_mut, dt * e0))

    etbl = etbl0.at[(0,) * pl.ndim].set(0.0)
    return pl_lift, etbl


def _Q_drift(
    dims, axes, coal_rates, aux
) -> tuple[GroupedKronProd, GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    i = list(axes).index
    return GroupedKronProd(
        [{i(pop): aux["drift"][pop] * coal_rates[pop]} for pop in axes], dims
    )


def _Q_mig_mut(
    t0, t1, dims, axes, mig_mat, aux, tr=False
) -> tuple[GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    i = list(axes).index
    Q_mut = GroupedKronProd([{i(pop): aux["mut"][pop]} for pop in axes], dims)

    # migration matrix is a bit trickier
    def f_Q_mig(t):
        if tr:
            M = mig_mat(t)
        else:
            M = mig_mat(-t + t1 + t0)
        terms = []
        for (s1, s2), m_ij in M.items():
            u1, u2 = aux["mig"][s1, s2]
            i1, i2 = map(list(axes).index, (s1, s2))
            if tr:
                u1, u2 = [{k: v.T for k, v in x.items()} for x in (u1, u2)]
            terms.append({i1: m_ij * u1[0], i2: u1[1]})
            terms.append({i2: m_ij * u2[1]})
        return GroupedKronProd(terms, dims)

    if tr:
        f_Q_mig.t = mig_mat.t
    else:
        f_Q_mig.t = -mig_mat.t + t1 + t0

    return f_Q_mig, Q_mut
