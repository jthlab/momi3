from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from momi3.common import Axes, PopCounter, Population, State
from momi3.math_functions import log_hypergeom

from .event import Event


@dataclass(frozen=True, kw_only=True)
class Downsample(Event):
    "downsample from m to n lineages (forwards in time)"
    pop: Population
    m: int
    n: int

    def _setup_impl(
        self, in_axes: Axes, ns: PopCounter
    ) -> tuple[Axes, PopCounter, dict]:
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

    def _execute_impl(self, st: State, params: dict, aux: dict) -> State:
        ((i, B),) = aux.items()
        plp = jnp.apply_along_axis(B.__matmul__, i, st.pl)
        return st._replace(pl=plp)
