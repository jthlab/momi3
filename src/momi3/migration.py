import itertools as it
import operator
from functools import reduce

import jax
import jax.numpy as jnp
import scipy.sparse as sps
from jax import jacfwd

from .common import Axes, Population
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


def lift_cm(params: dict, t: float, pl: jnp.ndarray, axes, aux):
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
    dims = pl.shape
    Q_mig, Q_drift, Q_mut = _Q_matrix(
        dims,
        axes,
        params["coal"],
        params["mig"],
        aux,
    )
    Q_lift = Q_mig + Q_drift * 0.5
    pl_lift = expmv(Q_lift.T, t, pl)
    assert pl_lift.shape == pl.shape
    D = reduce(operator.mul, pl.shape)
    # now compute the expected branch lengths
    e0 = jnp.eye(D, 1, 0).reshape(pl.shape)

    def f(theta):
        # note: Q_mut * (...) has to be implemented as multiplication from the right order for it to work with traced
        # jax code
        Q = Q_lift + Q_mut * theta
        v = expmv(Q, t, e0)
        return v

    etbl = jacfwd(f)(0.0).at[(0,) * pl.ndim].set(0.0)
    return pl_lift, etbl


def _Q_matrix(
    dims, axes, coal_rates, mig_mat, aux
) -> tuple[GroupedKronProd, GroupedKronProd, GroupedKronProd]:
    """construct Q matrix for continuously migrating populations"""
    s = list(coal_rates)  # these are the populations participating in the migration
    i = list(axes).index
    Q_drift = GroupedKronProd(
        [{i(ss): aux["drift"][ss] * coal_rates[ss]} for ss in s], dims
    )
    # ^^^^^^^^^^^^
    # This is *not* the same as
    #   Q_drift = KronProd([{i: aux['drift'][i] * (coal_rates[i] / 4.) for i in pop_inds}]):
    #   ^^^^^ is kronecker product, the other/correct is Kronkecker sum
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
    return Q_mig, Q_drift, Q_mut


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
