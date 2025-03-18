from typing import NamedTuple

import jax

from momi3.common import Axes


class State(NamedTuple):
    p: jax.Array
    s: float
    c: float
    t: float
    terminal: bool

    # p is an (d,)*n  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    # c is the probability that the first coalescence event has not occured by time t.

    def check_shape(self, ax: Axes) -> None:
        d = next(iter(ax.values())) - 1
        assert all(v == d + 1 for v in ax.values())
        # assert all(s == d + 1 for s in self.p.shape)
