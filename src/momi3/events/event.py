from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import os

import numpy as np
from jax import numpy as jnp

from momi3.common import Axes, PopCounter, Population, State, T
from momi3.math_functions import log_hypergeom


@dataclass(frozen=True, kw_only=True)
class Event:
    "Base class for events."
    bounds: Axes = None

    def setup(
        self, axes: Axes, ns: Counter[Population, int]
    ) -> tuple[Axes, PopCounter, T]:
        ax, ns, aux = self._setup_impl(axes, ns)
        if self.bounds:
            aux["bounds"] = {}
            assert set(ax) <= set(self.bounds)
            for pop, dim in ax.items():
                n = dim - 1
                assert self.bounds[pop] <= n
                if self.bounds[pop] < n:
                    ax, ns, aux["bounds"][pop] = Upsample(
                        pop=pop, m=self.bounds[pop]
                    ).setup(ax, ns)
        return ax, ns, aux

    def execute(self, st: State, params: dict, aux: T) -> State:
        st = self._execute_impl(st, params, aux)
        if os.environ.get('MOMI_PRINT_EVENTS'):
            print(self)
        if self.bounds:
            for pop in aux["bounds"]:
                st = Upsample(pop=pop, m=self.bounds[pop]).execute(
                    st, params, aux["bounds"][pop]
                )
        return st


@dataclass(frozen=True, kw_only=True)
class Upsample(Event):
    "upsample from m lineages (forwards in time)"
    pop: Population
    m: int

    def setup(self, in_axes: Axes, ns: PopCounter) -> tuple[Axes, PopCounter, dict]:
        n = in_axes[self.pop] - 1
        assert self.m <= n
        if self.m == n:
            return in_axes, ns, None
        i, j = np.ogrid[: n + 1, : self.m + 1]
        B = np.exp(log_hypergeom(M=n, N=self.m, n=i, k=j))
        Bplus = np.linalg.pinv(B, rcond=1e-5)
        nsp = deepcopy(ns)
        nsp[self.pop] = {self.pop: self.m}
        out_axes = deepcopy(in_axes)
        assert out_axes[self.pop] == n + 1
        out_axes[self.pop] = self.m + 1
        i = list(in_axes).index(self.pop)
        return out_axes, nsp, {i: Bplus}

    def execute(self, st: State, params: dict, aux: dict) -> State:
        if aux is None:
            # no bounding was possible/necessary, so setup set aux to None.
            return st
        ((i, Bplus),) = aux.items()
        pl_inds = list(range(st.ndim))
        out_inds = list(pl_inds)
        j = st.ndim
        out_inds[i] = j
        plp = jnp.tensordot(st.pl, tuple(pl_inds), Bplus, (j, i), tuple(out_inds))
        return st._replace(pl=plp)
