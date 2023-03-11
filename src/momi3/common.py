"miscellaneous shared functions that don't fit anywhere else"
from collections import Counter, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from secrets import token_hex
from typing import NamedTuple, OrderedDict, Sequence, TypeVar

import opt_einsum
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.util import safe_zip

oe_einsum = partial(opt_einsum.contract, optimize="optimal", backend="jax")


def Ne_t(Ne0, Ne1, t0, t1, t):
    return Ne0 * (Ne1 / Ne0) ** ((t1 - t) / (t1 - t0))


def traverse(params, path):
    for i in path:
        params = params[i]
    return params


def unique_strs(q: Sequence[str], k: int = 1, ell: int = 8) -> list[str]:
    "return a unique string of length l which is not in q"
    ret = []
    while len(ret) < k:
        s = token_hex(ell)
        if s not in q and s not in ret:
            ret.append(s)
    return ret


def unique_str(q, ell: int = 8):
    return unique_strs(q, 1, ell)[0]


# Some type aliases that are used throughout
Population = str
PopCounter = dict[
    Population, dict[Population, int]
]  # Maps populations to the populations they are ancestral to, and the sample size of each
Block = frozenset[Population]


@register_pytree_node_class
class Axes(OrderedDict[Population, int]):
    """An ordered mapping of populations to axis sizes."""

    def new_unique(self, k: int, ell: int = 8) -> list[Population]:
        return unique_str(self.keys(), k, ell)

    def tree_flatten(self):
        return (list(self.values()), list(self.keys()))

    @classmethod
    def tree_unflatten(cls, keys, values):
        return OrderedDict(safe_zip(keys, values))


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


T = TypeVar("T")


@dataclass
class Event:
    "Base class for events."

    def setup(
        self, axes: Axes, ns: Counter[Population, int]
    ) -> tuple[Axes, PopCounter, T]:
        return axes, ns, None

    def execute(self, st: State, params: dict, aux: T) -> State:
        return st


NoOp = Event


@dataclass
class Rename(Event):
    "Rename a population."
    old: Population
    new: Population

    def setup(
        self, in_axes: Axes, ns: Counter[Population, int]
    ) -> tuple[Axes, PopCounter, T]:
        out_axes = deepcopy(in_axes)
        out_axes[self.new] = out_axes.pop(self.old)
        ns = deepcopy(ns)
        ns[self.new] = ns.pop(self.old)
        return out_axes, ns, None

    def execute(self, st: State, params: dict, aux: T) -> State:
        return st


TimeTuple = namedtuple("TimeTuple", "t path")


class Time(TimeTuple):
    def __init__(self, t: float, path: tuple):
        super().__init__()

    def __str__(self):
        return f"Time({self.t})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.t)

    def __eq__(self, other: "Time") -> bool:
        assert isinstance(other, Time)
        return self.t == other.t

    def __lt__(self, other: "Time") -> bool:
        assert isinstance(other, Time)
        return self.t < other.t
