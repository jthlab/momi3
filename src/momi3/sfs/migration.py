import diffrax as dfx
import jax
import jax.numpy as jnp
import itertools as it
import scipy.sparse as sps
from loguru import logger
from typing import NamedTuple, Any
import jax.experimental.sparse as jesp
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
                mask = (t_start <= t) & (t < t_end)
                return mask.dot(r)
                # return jnp.where((t_start[j] <= t) & (t < t_end[j]), r[j], 0.0)

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
    pops = list(axes.keys())
    tm["drift"] = {pop: _drift(axes[pop] - 1) for pop in pops}
    tm["mut"] = {pop: _mutation(axes[pop] - 1) for pop in pops}
    tm["mig"] = {
        (p1, p2): _migration((axes[p1] - 1, axes[p2] - 1))
        for p1, p2 in it.product(pops, repeat=2)
    }

    # convert sparse matrices from scipy to JAX format
    def f(A):
        if isinstance(A, sps.spmatrix):
            ret = jesp.BCOO.from_scipy_sparse(A).sort_indices()
            ret.unique_indices = True
            return ret
        return A

    tm = jax.tree.map(f, tm)
    tm["axes"] = axes
    return tm


def _e0_like(pl):
    return jnp.zeros(pl.size).at[0].set(1.0).reshape(pl.shape)


def lift_cm(
    params: dict,
    t: tuple[float, float],
    pl: jnp.ndarray,
    axes,
    aux,
    involved_pops,
    const,
):
    if False:
        logger.debug("using sparse matrix exponentiation for {}", aux)
        f = _lift_cm_exp
    else:
        logger.debug("using diffeq solver for {}", aux)
        f = _lift_cm_exp
    return f(params, t, pl, axes, aux, involved_pops)


def _A(s, y, args):
    f_Q_mig, Q_mut, Q_drift, axes, tangent_sh, aux, etas, t = args

    def Qd(s):
        coal = {}
        for pop in axes:
            i = list(axes).index(pop)
            coal[i] = 1 / (4 * etas[pop](s, _no_searchsorted=True))
        new_A = []
        for Ai in Q_drift.A:
            assert len(Ai) == 1
            ((i, Aij),) = Ai.items()
            new_A.append({i: coal[i] * Aij})
        return Q_drift._replace(A=new_A)

    # backwards in time: t0 -> t1, solve for partial likelihood
    Q_mig = f_Q_mig(s)
    # multiply each entry of the drift tensor by the coalescent rate
    Q0 = Q_mig + Qd(s)
    r0 = Q0.T @ y[0]

    # forwards in time: t1 -> t0, solve for expected branch length
    sp = t[1] + t[0] - s
    Q_mig = f_Q_mig(sp)
    Q1 = Q_mig + Qd(sp)
    Q1 = Q1._replace(dims=tangent_sh)  # non-participating axes replaced by singletons
    Q_mut = Q_mut._replace(
        dims=tangent_sh
    )  # non-participating axes replaced by singletons
    r1 = Q1 @ y[1] + Q_mut @ y[2]
    r2 = Q1 @ y[2]
    return (r0, r1, r2)


def _lift_cm_exp(params, t, pl, axes, aux, involved_pops):
    # population sizes are changing, so we have to use a differential
    # equation solver
    dims = pl.shape
    assert all(
        a >= 5 for a in dims
    ), "dimensions too small for migration, require n >= 4 for all samples"
    etas = params["etas"]
    Q_drift = _Q_drift(dims, axes, {p: 1.0 for p in etas}, aux)
    f_Q_mig, Q_mut = _Q_mig_mut(t[0], t[1], dims, axes, params["mig"], aux)
    solver = dfx.Tsit5()
    term = dfx.ODETerm(_A)
    jump_ts = jnp.array([ti for eta in etas.values() for ti in eta.t])
    jump_ts = jnp.append(jump_ts, f_Q_mig.t)
    jump_ts = jnp.concatenate([jump_ts, t[1] + t[0] - jump_ts])
    jump_ts = jnp.sort(jump_ts)
    ssc = dfx.PIDController(jump_ts=jump_ts, rtol=1e-8, atol=1e-8)

    # compute d/dtheta x(t,theta)|{theta=0} using the forward sensitivity method.
    # we have x'(t, theta) = Q(t, theta) @ x(t, theta) and therefore
    # d/dtheta x'(t, theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    # = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # d/dt dx(t,theta)/dtheta d/dtheta x'(t,theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    #   = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # dF/dtheta = d(Q @ x)/dtheta = (Q_mut @ x)
    # the initial condition is d/dtheta(x(0, theta)) = 0.; x(0,theta) = e0
    # for computing branch length, we only need to track the populations that are involved in the migration
    sh = tuple(
        [pl.shape[i] if pop in involved_pops else 1 for i, pop in enumerate(axes)]
    )
    z = jnp.zeros(sh)
    e0 = z.at[(0,) * z.ndim].set(1.0)

    args = (f_Q_mig, Q_mut, Q_drift, axes, sh, aux, etas, t)
    y0 = (pl, z, e0)
    # FIXME this will lead to bugs
    # tangent_args += tuple([X._replace(dims=sh) for X in (Q_mut, Q_drift)])
    res = dfx.diffeqsolve(
        term,
        solver,
        t0=t[0],
        t1=t[1],
        dt0=(t[1] - t[0]) / 100.0,
        # dt0=None,
        y0=y0,
        args=args,
        stepsize_controller=ssc,
        # max_steps=4096,
        # max_steps=65536,
        # adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=10),
        # adjoint=dfx.BacksolveAdjoint(),
        # adjoint=dfx.DirectAdjoint(),
    )
    plp = res.ys[0][0]
    etbl = res.ys[1][0]
    inds = tuple([slice(None) if pop in involved_pops else 0 for pop in axes])
    etbl = etbl[inds]
    for x in (0, -1):
        etbl = etbl.at[(x,) * pl.ndim].set(0.0)
    return plp, etbl


# def _lift_cm_const(params: dict, t: tuple[float, float], pl: jnp.ndarray, axes, aux):
#     """
#     Lift partial likelihoods under continuous migration.
#
#     Args:
#         params: dict of parameters for migration model, one per population
#         t: length of time to lift
#         pl: partial likelihoods
#         axes: dict mapping populations to be lifted to their positions in pl
#         aux: the output of lift_cm_aux (see below)
#
#     Returns:
#         Tuple (lifted_likelihood, phi) where phi contains the expected branch lengths subtending each entry of the JSFS
#         for these populations.
#     """
#     dt = t[1] - t[0]
#     dims = pl.shape
#     s = (t[0] + t[1]) / 2.0
#     coal = {
#         pop: 1.0 / (4 * params["etas"][pop](s, _no_searchsorted=True)) for pop in axes
#     }
#     Q_drift = _Q_drift(dims, axes, coal, aux)
#
#     Q_drift1 = _Q_drift(dims, axes, {pop: 1.0 for pop in axes}, aux)
#
#     f_Q_mig, Q_mut = _Q_mig_mut(t[0], t[1], dims, axes, params["mig"], aux, tr=False)
#     Q_mig = f_Q_mig(s)
#     Q_mut = Q_mut
#     A = Q_drift.materialize() + Q_mig.materialize()
#     plf = pl.reshape(-1)
#     assert plf.shape == (A.shape[0],)
#     pl_lift = expm_unif(dt, A.T, plf).reshape(pl.shape)
#     assert pl_lift.shape == pl.shape
#
#     # # now compute the expected branch lengths
#     # class B:
#     #     def __matmul__(self, v):
#     #         v0, v1 = v
#     #         # equals the block matrix  [A, Q_mut; 0, A]
#     #         return A @ v0 + Q_mut @ v1, A @ v1
#
#     e0 = _e0_like(pl)
#
#     # z = jnp.zeros_like(e0)
#     # qm = _q_max(A + Q_mut)
#     # res = expm_unif(dt, B(), (z, e0), q_max=qm)
#
#     def f(theta):
#         A = Q_drift.materialize() + Q_mig.materialize() + theta * Q_mut.materialize()
#         q_max = abs(A.data).max()
#         return expm_unif(dt, A, e0.reshape(-1), q_max=q_max).reshape(pl.shape)
#
#     xxx = f(0.0)
#     etbl = jacrev(f)(0.0)
#
#     res = _lift_cm_exp(params, t, pl, axes, aux)
#
#     # another option
#     A = Q_drift.materialize() + Q_mig.materialize()
#     B = Q_mut.materialize()
#     Z = jesp.empty(A.shape)
#     block0 = jesp.bcoo_concatenate([A, B], dimension=1)
#     block1 = jesp.bcoo_concatenate([Z, A], dimension=1)
#     block = jesp.bcoo_concatenate([block0, block1], dimension=0)
#
#     import numpy as np
#
#     vf = jnp.concatenate([0 * e0.reshape(-1), e0.reshape(-1)])
#     v = (0 * e0, e0)
#
#     dt = 0.01
#     for i in range(int((t[1] - t[0]) / dt)):
#         dv = _A(s, v, (f_Q_mig, Q_mut, Q_drift1, dims, axes, aux, params["etas"]))
#         v = (v[0] + dt * dv[0], v[1] + dt * dv[1])
#         dvf = block @ vf
#         vf = vf + dt * dvf
#         print(
#             i,
#             np.linalg.norm(vf[:77] - v[0].reshape(-1)),
#             np.linalg.norm(vf[77:] - v[1].reshape(-1)),
#         )
#
#     etbl = etbl.at[(0,) * pl.ndim].set(0.0)
#
#     return pl_lift, etbl


def _Q_drift(
    dims, axes, coal_rates, aux
) -> tuple[GroupedKronProd, GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    i = list(axes).index
    return GroupedKronProd(
        [{i(pop): aux["drift"][pop] * coal_rates[pop]} for pop in axes], dims
    )


def _Q_mig_mut(
    t0, t1, dims, axes, mig_mat, aux
) -> tuple[GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    i = list(axes).index
    Q_mut = GroupedKronProd([{i(pop): aux["mut"][pop]} for pop in axes], dims)

    # migration matrix is a bit trickier
    def f_Q_mig(t):
        M = mig_mat(t)
        terms = []
        for (s1, s2), m_ij in M.items():
            u1, u2 = aux["mig"][s1, s2]
            i1, i2 = map(list(axes).index, (s1, s2))
            terms.append({i1: m_ij * u1[0], i2: u1[1]})
            terms.append({i2: m_ij * u2[1]})
        return GroupedKronProd(terms, dims)

    f_Q_mig.t = mig_mat.t
    return f_Q_mig, Q_mut
