from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import lax
from jax import numpy as jnp

from momi3.common import Axes, PopCounter, Population, State
from momi3.math_functions import log_hypergeom


@dataclass
class Downsample:
    "downsample from m to n lineages (forwards in time)"
    pop: Population
    m: int
    n: int

    def __hash__(self):
        return hash((self.m, self.n, self.pop))

    def setup(self, in_axes: Axes, ns: PopCounter) -> tuple[Axes, PopCounter, dict]:
        i, j = np.ogrid[: self.m + 1, : self.n + 1]
        B = jnp.exp(log_hypergeom(M=self.m, N=self.n, n=i, k=j))
        assert ns[self.pop] == {self.pop: self.n}
        nsp = deepcopy(ns)
        nsp[self.pop] = {self.pop: self.m}
        out_axes = deepcopy(in_axes)
        assert out_axes[self.pop] == self.n + 1
        out_axes[self.pop] = self.m + 1
        i = list(in_axes).index(self.pop)
        return out_axes, nsp, {i: B}

    def execute(self, st: State, params: dict, aux: dict) -> State:
        ((i, B),) = aux.items()
        plp = jnp.apply_along_axis(B.__matmul__, i, st.pl)
        return st._replace(pl=plp)


@dataclass
class Upsample:
    "upsample from m lineages (forwards in time)"
    pop: Population
    m: int

    def setup(self, in_axes: Axes, ns: PopCounter) -> tuple[Axes, PopCounter, dict]:
        n = in_axes[self.pop] - 1
        assert self.m <= n
        if self.m == n:
            return in_axes, ns, None
        i, j = np.ogrid[: n + 1, : self.m + 1]
        B = jnp.exp(log_hypergeom(M=n, N=self.m, n=i, k=j))
        Q, R = jnp.linalg.qr(B)
        nsp = deepcopy(ns)
        nsp[self.pop] = {self.pop: self.m}
        out_axes = deepcopy(in_axes)
        assert out_axes[self.pop] == n + 1
        out_axes[self.pop] = self.m + 1
        i = list(in_axes).index(self.pop)
        return out_axes, nsp, {i: (Q, R)}

    def execute(self, st: State, params: dict, aux: dict) -> State:
        if aux is None:
            # no bounding was possible/necessary, so setup set aux to None.
            return st
        ((i, (Q, R)),) = aux.items()

        def f(x):
            return lax.linalg.triangular_solve(R, Q.T @ x, left_side=True, lower=False)

        plp = jnp.apply_along_axis(f, i, st.pl)
        return st._replace(pl=plp)
