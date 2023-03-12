import itertools as it

import diffrax as dfx
import jax
import jax.numpy as jnp
import scipy.sparse as sps
from jax import jacfwd, lax, vmap

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
        lambda A: _bcoo_from_sp(A) if isinstance(A, sps.spmatrix) else A, tm
    )
    tm["axes"] = axes
    return tm


def _e0_like(pl):
    return jnp.zeros(pl.size).at[0].set(1.0).reshape(pl.shape)


def lift_cm(params: dict, t: tuple[float, float], pl: jnp.ndarray, axes, aux):
    Ne = params["Ne"]
    const = all(not isinstance(Ne[pop], tuple) for pop in Ne)
    if const:
        f = _lift_cm_const
    else:
        f = _lift_cm_exp
    return f(params, t, pl, axes, aux)


def _lift_cm_exp(params, t, pl, axes, aux):
    # population sizes are changing, so we have to use a differential
    # equation solver
    dims = pl.shape
    Ne = params["Ne"]
    Q_mig, Q_mut = _Q_mig_mut(
        dims,
        axes,
        params["mig"],
        aux,
    )

    def A(s, y, args):
        theta, tr = args
        coal = {}
        for pop in Ne:
            if isinstance(Ne[pop], tuple):
                N1, N0 = Ne[pop]
                coal[pop] = 1.0 / Ne_t(N0, N1, t[0], t[1], s)
            else:
                # Ne is a float
                coal[pop] = 1.0 / Ne[pop]
        Qd = _Q_drift(dims, axes, coal, aux)
        Q_lift = Q_mig + Qd * 0.5 + Q_mut * theta
        # after passage through lax.cond dims is traced, but we need it staticially known to compute the matmul
        T = lax.cond(tr, lambda x: x.T, lambda x: x, Q_lift)._replace(dims=Q_lift.dims)
        return T @ y

    term = dfx.ODETerm(A)
    # solver = dfx.ImplicitEuler(nonlinear_solver=dfx.NewtonNonlinearSolver(rtol=1e-3, atol=1e-6))
    solver = dfx.Dopri5()

    def solve(theta, y0, tr):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[1],
            dt0=(t[1] - t[0]) / 50,
            y0=y0,
            args=(theta, tr),
        ).ys[0]

    eps = 1e-7
    e0 = _e0_like(pl)
    res = vmap(solve, (0, 0, 0))(
        jnp.array([0.0, eps, -eps]),
        jnp.array([pl, e0, e0]),
        jnp.array([True, False, False]),
    )
    plp = res[0]
    # etbl = jacrev(solve)(0.)
    # got an error getting this to work with sparse matrices, so for now settle for finite diffs
    etbl = (res[1] - res[2]) / (2 * eps)
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
    Q_lift = (Q_mig + Q_drift * 0.5) * dt
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
    dims, axes, mig_mat, aux
) -> tuple[GroupedKronProd, GroupedKronProd, GroupedKronProd]:
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
            i1 = i(s1)
            i2 = i(s2)
            terms.append({i1: m_ij * u1[0], i2: u1[1]})
            terms.append({i2: m_ij * u2[1]})
    Q_mig = GroupedKronProd(terms, dims)
    return Q_mig, Q_mut


def _bcoo_kron(A, B):
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, empty

    assert A.ndim == B.ndim == 2
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nse == 0 or B.nse == 0:
        return empty(output_shape)

    # expand entries of a into blocks
    A_row, A_col = A.indices.T
    B_row, B_col = B.indices.T
    row = A_row.repeat(B.nse)
    col = A_col.repeat(B.nse)
    data = A.data.repeat(B.nse)

    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row, col = row.reshape(-1, B.nse), col.reshape(-1, B.nse)
    row += B_row
    col += B_col
    row, col = row.reshape(-1), col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, B.nse) * B.data
    data = data.reshape(-1)

    (a,) = data.nonzero()
    inds = jnp.stack([row, col], axis=1)

    return BCOO((data[a], inds[a]), shape=output_shape)


def _bcoo_kronsum(A, B):
    from jax.experimental.sparse import eye

    I_A = eye(A.shape[0])
    I_B = eye(B.shape[0])
    return _bcoo_kron(A, I_B) + _bcoo_kron(I_A, B)


def _bcoo_from_sp(A):
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    A = A.tocoo()
    return BCOO((A.data, jnp.stack([A.row, A.col], axis=1)), shape=A.shape)
