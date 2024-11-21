from typing import Any

import demes
import jax
import jax.numpy as jnp

from momi3.common import Axes
from momi3.event_tree import EventTree

from . import events
from .state import State


class IicrEventTree(EventTree):
    def __init__(self, demo: demes.Graph, num_samples: dict[str, int]):
        super().__init__(demo, num_samples, events)

    def _init_leaves(self):
        super()._init_leaves()
        for deme in self._demo.demes:
            pop = deme.name
            n = self._num_samples.get(pop, 0)
            self.nodes[self.leaves[pop]].update(
                {"axes": Axes({pop: n + 1}), "ns": {pop: {pop: n}}}
            )

    def execute(self, params: dict, t: float, aux: Any) -> jax.Array:
        for pop in self.leaves:
            n = self._num_samples.get(pop, 0)
            sh = (1,) * n
            p = jnp.ones(sh)
            self.nodes[self.leaves[pop]]["state"] = State(
                p=p, s=1.0, c=0.0, t=t, terminal=False
            )

        ret = super().execute(params, aux)
        return (ret.c, ret.s)
