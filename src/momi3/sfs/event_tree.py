import demes
import jax
import jax.numpy as jnp

import momi3.sfs.events as events
from momi3.event_tree import Axes, EventTree, Population

from .state import State


class SfsEventTree(EventTree):
    def __init__(self, demo: demes.Graph, num_samples: dict[str, int]):
        self.num_samples = num_samples
        super().__init__(demo, events)

    def _init_tree(self):
        super()._init_tree()
        for j, deme in enumerate(self._demo.demes):
            node = self.leaves[deme.name]
            ns = self.num_samples.get(deme.name, 0)
            if ns < 4:
                # for continuous migration, we require that there are at least four nodes. so for now we just enforce
                # this globally. slightly wasteful if there is not any cm 🤷.
                v = self.node_like(node)
                self.add_edge(
                    node, v, event=events.Downsample(pop=deme.name, m=4, n=ns)
                )

    def _setup(self):
        for deme in self._demo.demes:
            pop = deme.name
            n = self.num_samples.get(pop, 0)
            self.nodes[self.leaves[pop]].update(
                {"axes": Axes({pop: n + 1}), "ns": {pop: {pop: n}}}
            )
        return super()._setup()

    def execute(
        self, params: dict, leaf_state: dict[Population, jnp.ndarray], auxd: dict
    ) -> jnp.ndarray:
        for pop in self._leaves:
            # int partial likelihoods causes all sorts of problems further down
            ns = self.num_samples.get(pop, 0)
            X = leaf_state.get(pop, jax.nn.one_hot(jnp.array([0]), ns + 1)[0]).astype(
                float
            )
            assert X.shape == (ns + 1,)
            l0 = (X[0] == 1.0).astype(float)  # & (X[pop][1:] == 0.0).all()
            self.nodes[self._leaves[pop]]["state"] = State(pl=X, phi=0.0, l0=l0)

        return super().execute(params, auxd)
