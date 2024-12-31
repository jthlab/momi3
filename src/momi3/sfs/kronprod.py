import operator
from functools import reduce, singledispatch
from typing import Union

import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, empty, eye, sparsify


@jdc.pytree_dataclass
class KronProd:
    """Class representing a matrix A defined by a sum of Kronecker products: A = ∑_n ⊗_i A_{ni}.

    Params:
        dims: dimensions of each axis.
        A: list of dicts containing the entries {i: A_{ni}} shown above. Missing entries are equal to the
           identity matrix.

    Notes:
        Every matrix is assumed to be square. Everything is assumed to be conformable, same dims, etc.
    """

    A: list[dict[int, Union[jnp.ndarray, BCOO]]]
    dims: jdc.Static[tuple[int, ...]]

    def _replace(self, *args, **kwargs):
        return jdc.replace(self, *args, **kwargs)

    def _check_dims(self):
        for Ai in self.A:
            for i in Ai:
                d = self.dims[i]
                assert Ai[i].shape == (d, d)

    @classmethod
    def eye(cls, dims):
        return cls([], dims)

    @property
    def dtype(self):
        return list(self.A[0].values())[0].dtype

    def trace(self) -> float:
        ret = 0.0
        for Ai in self.A:
            r = 1.0
            for i, d in enumerate(self.dims):
                if i in Ai:
                    r *= tr(Ai[i])
                else:
                    r *= d
            ret += r
        return ret

    def bound_norm1(self) -> float:
        ret = 0.0
        for Ai in self.A:
            ret += reduce(operator.mul, [norm1(x) for x in Ai.values()])
        return ret

    def materialize(self) -> BCOO:
        """Return a sparse matrix representation of A."""

        def ident():
            return list(map(eye, self.dims))

        D = reduce(operator.mul, self.dims)
        ret = empty((D, D))
        for An in self.A:
            mats = ident()
            for i in An:
                mats[i] = An[i]
            ret += reduce(kron, mats)
        return ret

    def todense(self):
        """Convert from sparse to dense representation of Ai"""
        return self.__class__(
            [{k: v.todense() for k, v in d.items()} for d in self.A], self.dims
        )

    def __matmul__(self, x):
        """Compute the matrix vector product Ax."""
        ret = jnp.zeros_like(x)
        for An in self.A:
            y = x
            for i, Ani in An.items():
                y = sparsify(jnp.tensordot)(Ani, y, axes=([1], [i]))
                # "The shape of the result consists of the non-contracted axes of the first tensor, followed by the
                # non-contracted axes of the second." -- np docs
                y = jnp.moveaxis(y, 0, i)
                assert y.shape == x.shape
            ret += y
        return ret

    def _reduce(self) -> "KronProd":
        s = {}
        t = []
        for Ai in self.A:
            if len(Ai) == 1:
                ((k, v),) = Ai.items()
                if k in s:
                    s[k] += v
                else:
                    s[k] = v
            else:
                t.append(Ai)
        return self.__class__(t + [{i: s[i]} for i in s], self.dims)

    def __add__(self, other):
        if not isinstance(other, KronProd):
            return NotImplemented
        assert self.dims == other.dims
        return self._replace(A=self.A + other.A)._reduce()

    def __sub__(self, other):
        if not isinstance(other, KronProd):
            return NotImplemented
        assert self.dims == other.dims
        return self + (other.__mul__(-1))

    def __mul__(self, c):
        "multiply by constant."
        # (by multilinearity, only one/the first term in each summand gets multiplied)
        try:
            return self._replace(
                A=[
                    {
                        k: (c if i == 0 else 1.0) * v
                        for i, (k, v) in enumerate(Ai.items())
                    }
                    for Ai in self.A
                ]
            )
        except Exception:
            return NotImplemented

    def __rmul__(self, c):
        "multiply by constant."
        # (by multilinearity, only one/the first term in each summand gets multiplied)
        return self.__mul__(c)

    @property
    def T(self) -> "KronProd":
        return self._replace(A=[{k: v.T for k, v in Ai.items()} for Ai in self.A])


@jdc.pytree_dataclass
class GroupedKronProd(KronProd):
    """Group operations between pairs of indices."""

    def __old_matmul__(self, other: jnp.ndarray) -> jnp.ndarray:
        ret = jnp.zeros_like(other)
        d = len(self.dims)
        f = 1.0 / (d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                n = self.dims[i] * self.dims[j]
                Qij: BCOO = empty((n, n))
                for Ai in self.A:
                    assert len(Ai) <= 2
                    if Ai.keys() == {i, j}:
                        Qij += kron(Ai[i], Ai[j])
                    elif Ai.keys() == {i}:
                        Qij += kron(f * Ai[i], eye_like(Ai[i], self.dims[j]))
                    elif Ai.keys() == {j}:
                        Qij += kron(eye_like(Ai[j], self.dims[i]), f * Ai[j])
                try:
                    Qij = Qij.sort_indices()
                    assert Qij.indices_sorted
                except AttributeError:
                    assert isinstance(Qij, jnp.ndarray)

                r1 = other.swapaxes(i, 0).swapaxes(j, 1)
                ret += (
                    (Qij @ r1.reshape(n, -1))
                    .reshape(r1.shape)
                    .swapaxes(j, 1)
                    .swapaxes(
                        0,
                        i,
                    )
                )
        return ret

    def __matmul__(self, other):
        assert self.dims == other.shape
        ret = jnp.zeros_like(other)
        d = len(self.dims)
        other_inds = tuple(range(d))
        f = 1.0 / (d - 1)
        for Ai in self.A:
            k = list(Ai.keys())
            if len(k) == 2:
                i, j = k
                out_inds = list(other_inds)
                out_inds[i] = d + 1
                out_inds[j] = d + 2
                ret += jnp.einsum(
                    Ai[i],
                    (d + 1, i),
                    Ai[j],
                    (d + 2, j),
                    other,
                    other_inds,
                    tuple(out_inds),
                )
            else:
                assert len(k) == 1
                i = k[0]
                out_inds = list(other_inds)
                out_inds[i] = d + 1
                ret += jnp.einsum(
                    f * Ai[i], (d + 1, i), other, other_inds, tuple(out_inds)
                )
        return ret


@singledispatch
def eye_like(mat: jnp.ndarray, n: int) -> jnp.ndarray:
    return jnp.eye(n)


@eye_like.register
def _(mat: BCOO, n: int) -> BCOO:
    return eye(n)


@singledispatch
def tr(A: BCOO):
    # sparse matrix trace
    assert A.ndim == 2
    return jnp.sum(A.data * (A.indices[:, 0] == A.indices[:, 1]))


tr.register(KronProd, KronProd.trace)
tr.register(jnp.ndarray, jnp.trace)
tr.register(np.ndarray, np.trace)


@singledispatch
def kron(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    assert A.ndim == B.ndim == 2
    assert isinstance(A, jnp.ndarray) and isinstance(B, jnp.ndarray)
    return jnp.kron(A, B)


@kron.register
def _spkron(A: BCOO, B: BCOO) -> BCOO:
    assert isinstance(A, BCOO) and isinstance(B, BCOO)
    # sparse kronecker product of BCOO matrices. (actually just COO)
    assert A.ndim == B.ndim == 2
    return (
        (A[:, None, :, None] * B[None, :, None, :])
        .reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        .sort_indices()
    )


@singledispatch
def norm1(A: BCOO):
    assert A.ndim == 2
    return jnp.zeros(A.shape[0]).at[A.indices[:, 1]].add(abs(A.data)).max()


@norm1.register
def _(A: np.ndarray):
    return np.linalg.norm(A, 1)
