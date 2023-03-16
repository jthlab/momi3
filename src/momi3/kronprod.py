import operator
from functools import reduce, singledispatch
from typing import NamedTuple, Union

import numpy as np
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, empty, eye, sparsify


class KronProd(NamedTuple):
    """Class representing a matrix A defined by a sum of Kronecker products: A = ∑_n ⊗_i A_{ni}.

    Params:
        dims: dimensions of each axis.
        A: list of dicts containing the entries {i: A_{ni}} shown above. Missing entries are equal to the
           identity matrix.

    Notes:
        Every matrix is assumed to be square. Everything is assumed to be conformable, same dims, etc.
    """

    A: list[dict[int, Union[jnp.ndarray, BCOO]]]
    dims: tuple[int, ...]

    def _check_dims(self):
        for Ai in self.A:
            for i in Ai:
                d = self.dims[i]
                assert Ai[i].shape == (d, d)

    @classmethod
    def eye(cls, dims):
        return cls([], dims)

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
            ret += reduce(_spkron, mats)
        return ret

    def densify(self):
        """Convert from sparse to dense representation of Ai"""
        return self.__class__([{k: v.todense() for k, v in d.items()} for d in self.A])

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


class GroupedKronProd(KronProd):
    """Group operations between pairs of indices."""

    def __matmul__(self, other: jnp.ndarray) -> jnp.ndarray:
        ret = jnp.zeros_like(other)
        d = len(self.dims)
        f = 1.0 / (d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                n = self.dims[i] * self.dims[j]
                Qij = empty((n, n))
                for Ai in self.A:
                    assert len(Ai) <= 2
                    if Ai.keys() == {i, j}:
                        Qij += _spkron(Ai[i], Ai[j])
                    elif Ai.keys() == {i}:
                        Qij += _spkron(f * Ai[i], eye(self.dims[j]))
                    elif Ai.keys() == {j}:
                        Qij += _spkron(eye(self.dims[i]), f * Ai[j])
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


@singledispatch
def tr(A: BCOO):
    # sparse matrix trace
    assert A.ndim == 2
    return jnp.sum(A.data * (A.indices[:, 0] == A.indices[:, 1]))


tr.register(KronProd, KronProd.trace)
tr.register(jnp.ndarray, jnp.trace)
tr.register(np.ndarray, np.trace)


def _spkron(A, B) -> BCOO:
    # sparse kronecker product of BCOO matrices. (actually just COO)
    assert A.ndim == B.ndim == 2
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]
    )


@singledispatch
def norm1(A: BCOO):
    assert A.ndim == 2
    return jnp.zeros(A.shape[0]).at[A.indices[:, 1]].add(abs(A.data)).max()


@norm1.register
def _(A: np.ndarray):
    return np.linalg.norm(A, 1)


def _bcoo_to_sp(A):
    from scipy.sparse import coo_matrix

    assert A.ndim == 2
    return coo_matrix((A.data, A.indices.T), shape=A.shape)
