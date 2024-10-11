import operator
from functools import partial, reduce, singledispatch

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit, vmap
from jax.experimental.sparse import BCOO
from scipy.sparse.linalg import LinearOperator, expm_multiply

from momi3.kronprod import KronProd


@singledispatch
def expmv(A: np.ndarray | jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return expm_multiply(A, B)


@expmv.register
def expmv_sparse(A: BCOO, B: jnp.ndarray) -> jnp.ndarray:
    assert A.ndim == 2
    data = np.asarray(A.data)
    row, col = np.asarray(A.indices.T)
    A_sp = scipy.sparse.coo_matrix((data, (row, col)), shape=A.shape)
    return expm_multiply(A_sp, B)


@expmv.register
def expmv_kronprod(A: KronProd, B: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the matrix exponential of a Kronecker product matrix.

    Args:
        A: A multilinear operator
        B: vector

    Returns:
        The matrix exponential of A acting on B.
    """
    involved_dims = tuple([i for An in A.A for i in An])
    # the "uninvolved" dimensions are treated as batch dimensions. move those
    # to the front in B.
    # move the batch dimensions to the front
    batch_dims = tuple(set(range(B.ndim)) - set(involved_dims))
    # permutation to move the batch dimensions to the front
    pi = batch_dims + tuple(involved_dims)
    # First, "remap" the KronProd A to act on the permuted involved dims.
    A_pi = KronProd(
        A=[{involved_dims.index(i): Ai for i, Ai in An.items()} for An in A.A],
        dims=[A.dims[i] for i in involved_dims],
    )
    B_pi = B.transpose(pi)
    orig_shape = B.shape
    B_pi = B_pi.reshape(-1, reduce(operator.mul, (A.dims[i] for i in involved_dims)))
    # now B_pi is in the form of a matrix, with the batch dimensions as the first.
    # we're ready to call expm_multiply(A_pi, B_pi).
    Y = vmap(_expmv_kronprod, (None, 0))(A_pi, B_pi)
    # restore shape and move axes back to their original locations.
    Y = Y.reshape(orig_shape)
    # invert the permutation
    pi_inv = tuple(pi.index(i) for i in range(len(pi)))
    return Y.transpose(pi_inv)


@jit
def _matvec(A: KronProd, x: jnp.ndarray) -> jnp.ndarray:
    return (A @ x.reshape(A.dims)).reshape(x.shape)


class _KPLinOp(LinearOperator):
    def __init__(self, A: KronProd):
        self.A = A
        self.dtype = A.dtype
        self.shape = (reduce(operator.mul, A.dims),) * 2

    def _matvec(self, x):
        return _matvec(self.A, x)

    def _rmatvec(self, x):
        return _matvec(self.A.T, x)


@jax.custom_vjp
def _expmv_kronprod(A: KronProd, B: jnp.ndarray) -> jnp.ndarray:
    # A: KronProd, B: vector
    assert B.ndim == 1
    return _expmv_kronprod_impl(A, B)


def _expmv_kronprod_impl(A: KronProd, B: jnp.ndarray) -> jnp.ndarray:
    # A is a KronProd, B is a matrix
    def f(A, B, tr):
        A_linop = _KPLinOp(A)
        return expm_multiply(A_linop, B, traceA=tr)

    return jax.pure_callback(
        f, jax.ShapeDtypeStruct(B.shape, B.dtype), A, B, A.trace(), vectorized=False
    )


def _expmv_kronprod_fwd(A, B):
    expm_A_B = _expmv_kronprod(A, B)
    return expm_A_B, (A, B)


@jit
def _augmatvec(A: KronProd, gBt: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    n = reduce(operator.mul, A.dims)
    assert x.shape[0] == 2 * n
    x0, x1 = jnp.split(x, [n])
    # multiply block
    # [A gBt]
    # [0 A]

    @partial(vmap, in_axes=(1,), out_axes=1)
    def f(xi):
        return (A @ xi.reshape(A.dims)).reshape(xi.shape)

    return jnp.concatenate([f(x0) + gBt @ x1, f(x1)])


@jit
def _augrmatvec(A: KronProd, gBt: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    n = reduce(operator.mul, A.dims)
    assert x.shape[0] == 2 * n
    x0, x1 = jnp.split(x, [n])
    # multiply block
    # [A.T 0]
    # [gBt.T A.T]

    @partial(vmap, in_axes=(1,), out_axes=1)
    def f(xi):
        return (A.T @ xi.reshape(A.dims)).reshape(xi.shape)

    return jnp.concatenate([f(x0), gBt.T @ x0 + f(x1)])


class _KPLinOpAug(LinearOperator):
    def __init__(self, A: KronProd, B: jnp.ndarray, g: jnp.ndarray):
        self.A = A
        assert g.ndim == B.ndim == 1
        self.gBt = g[:, None] @ B[None, :]
        self.dtype = A.dtype
        self.shape = (2 * reduce(operator.mul, A.dims),) * 2

    def _matvec(self, x):
        return _augmatvec(self.A, self.gBt, x)

    def _rmatvec(self, x):
        return _augrmatvec(self.A, self.gBt, x)


def _expmv_kronprod_bwd(res, g):
    A, B = res
    assert B.ndim == g.ndim == 1
    n = reduce(operator.mul, A.dims)
    assert B.shape == g.shape == (n,)

    def f(A, B, g, tr):
        n = B.shape[0]
        A_aug_linop = _KPLinOpAug(A, B, g)
        X = np.concatenate([np.eye(n), np.zeros([n, n])])
        return expm_multiply(A_aug_linop, X, traceA=tr)[:n]

    grad_A = jax.pure_callback(
        f,
        jax.ShapeDtypeStruct((n, n), B.dtype),
        A,
        B,
        g,
        2 * A.trace(),
        vectorized=False,
    )

    # Compute gradient with respect to B
    grad_B = _expmv_kronprod(A, g)

    return grad_A, grad_B


_expmv_kronprod.defvjp(_expmv_kronprod_fwd, _expmv_kronprod_bwd)
