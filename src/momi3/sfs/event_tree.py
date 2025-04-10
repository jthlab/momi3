import jax
import jax.numpy as jnp

import momi3.sfs.events as events
from momi3.common import Axes
from momi3.event_tree import EventTree, Population

from .state import State


class SfsEventTree(EventTree):
    def __init__(self, demo, num_samples):
        self._num_samples = num_samples
        super().__init__(demo, events)

    def _init_leaves(self):
        # initialize leaf sample sizes
        super()._init_leaves()
        for deme in self._demo.demes:
            pop = deme.name
            n = self._num_samples.get(pop, 0)
            self.nodes[self.leaves[pop]].update(
                {"axes": Axes({pop: n + 1}), "ns": {pop: {pop: n}}}
            )

        # add downsample events where necessary
        # for continuous migration, we require that there are at least four nodes.
        # so for now we just enforce this globally. slightly wasteful if there is
        # not any cm 🤷.
        # now proceed with usual setup
        for j, deme in enumerate(self._demo.demes):
            node = self.leaves[deme.name]
            ns = self._num_samples.get(deme.name, 0)
            if ns < 4:
                v = self.node_like(
                    node, event=events.Downsample(pop=deme.name, m=4, n=ns)
                )
                self.add_edge(node, v)

    def execute(
        self, params: dict, leaf_state: dict[Population, jnp.ndarray], aux: dict
    ) -> jnp.ndarray:
        for pop in self._leaves:
            # int partial likelihoods causes all sorts of problems further down
            ns = self._num_samples.get(pop, 0)
            X = leaf_state.get(pop, jax.nn.one_hot(jnp.array([0]), ns + 1)[0]).astype(
                float
            )
            assert X.shape == (ns + 1,)
            l0 = (X[0] == 1.0).astype(float)  # & (X[pop][1:] == 0.0).all()
            self.nodes[self._leaves[pop]]["state"] = State(
                pl=X, phi=0.0, l0=l0, terminal=False
            )

        return super().execute(params, aux)
