from itertools import count
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
        i = count(-1)
        for pop in self.leaves:
            # idea here is that the state is a (d+1, d+1, ..., d+1)-tensor where
            # T[i, j, ..., k] is the probability that lineage 1 is in deme i, lineage 2 is in deme j, etc.
            # deme d+1 is a special deme that represents the "outside" deme
            k = [1] * self._n
            for p in num_samples:
                for j in range(num_samples.get(p, 0)):
                    if p == pop:
                        k[next(i)] = 0
            p = jnp.zeros((2,) * self._n).at[tuple(k)].set(1.0)
            self.nodes[self.leaves[pop]]["state"] = State(
                p=p, s=1.0, c=0.0, t=t, terminal=False
            )

        ret = super().execute(params=params, auxd=aux)
        return (ret.c, ret.s)
