from typing import Any

import demes
import jax
import jax.numpy as jnp

from momi3.common import Axes
from momi3.event_tree import EventTree

from . import events
from .state import State


class IicrEventTree(EventTree):
    def __init__(self, demo: demes.Graph, n: int):
        self._n = n
        super().__init__(demo, events)

    def _init_leaves(self):
        super()._init_leaves()
        for deme in self._demo.demes:
            pop = deme.name
            self.nodes[self.leaves[pop]].update(
                {"axes": Axes({pop: self._n}), "ns": {pop: {pop: self._n}}}
            )

    def execute(self, num_samples: dict, params: dict, t: float, aux: Any) -> jax.Array:
        i = -1
        for pop in self.leaves:
            # idea here is that the state is a (d+1, d+1, ..., d+1)-tensor where
            # T[i, j, ..., k] is the probability that lineage 1 is in deme i, lineage 2 is in deme j, etc.
            # deme d+1 is a special deme that represents the "outside" deme
            k = jnp.ones(self._n, dtype=jnp.int32)
            for p in num_samples:
                for j in range(self._n):
                    accept = (p == pop) & (j < num_samples[p])
                    i = jnp.where(accept, i + 1, i)
                    k1 = k.at[i].set(0)
                    k = jnp.where(accept, k1, k)
            p = jnp.zeros((2,) * self._n).at[tuple(k)].set(1.0)
            self.nodes[self.leaves[pop]]["state"] = State(
                p=p, s=1.0, c=0.0, t=t, terminal=False
            )

        ret = super().execute(params=params, auxd=aux)
        return (ret.c, ret.s)
