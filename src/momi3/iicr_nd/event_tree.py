from typing import Any

import demes
import jax
import jax.numpy as jnp

from momi3.common import Axes
from momi3.event_tree import EventTree

from . import events
from .state import State


class IicrNdEventTree(EventTree):
    def __init__(self, demo: demes.Graph, n: int):
        self._n = n
        super().__init__(demo, events)

    def _init_leaves(self):
        super()._init_leaves()
        n = self._n
        for deme in self._demo.demes:
            pop = deme.name
            self.nodes[self.leaves[pop]].update(
                {"axes": Axes({pop: n + 1}), "ns": {pop: {pop: n}}}
            )

    def execute(
        self, params: dict, num_samples: dict[str, int], t: float, aux: Any
    ) -> jax.Array:
        # assert sum(num_samples.values()) == self._n
        I = jnp.eye(self._n + 1)  # noqa: E741
        for pop in self.leaves:
            p = I[num_samples.get(pop, 0)]
            self.nodes[self.leaves[pop]]["state"] = State(
                p=p, s=1.0, c=0.0, t=t, terminal=False
            )
        ret = super().execute(params, aux)
        return (ret.c, ret.s)
