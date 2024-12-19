from typing import NamedTuple

import jax.numpy as jnp

from momi3.common import Axes


class State(NamedTuple):
    """The state of a node in the event tree:

    Attributes:
        pl: the likelihood of the subtended leaf alleles conditional on the number of derived alleles at this node
        phi: the total expected branch length subtending the leaf alleles
        l0: do the leaves beneath this pl all have zero derived alleles?
    """

    pl: jnp.ndarray
    phi: float
    l0: bool
    terminal: bool

    def check_shape(self, axes: Axes):
        assert self.pl.shape == tuple(axes.values())
