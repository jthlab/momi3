"miscellaneous shared functions that don't fit anywhere else"

from secrets import token_hex
from typing import OrderedDict, Sequence, NamedTuple

from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.util import safe_zip

oe_einsum = jnp.einsum


# Some type aliases that are used throughout
Population = str
PopCounter = dict[
    Population, dict[Population, int]
]  # Maps populations to the populations they are ancestral to, and the sample size of each
Block = frozenset[Population]
Path = tuple[str | int, ...]


def get_path(params, path: Path) -> float:
    for i in path:
        params = params[i]
    return params


def set_path(params, path: Path, value: float):
    for i in path[:-1]:
        params = params[i]
    params[path[-1]] = value


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

    @property
    def n(self):
        return sum(a - 1 for a in self.values())


class Time(NamedTuple):
    t: float
    path: Path


def inv_softplus(y):
    return y + jnp.log1p(-jnp.exp(-y))


def inv_softmax(y):
    "softmax(inv_softmax(softmax(y))) = softmax(y)"
    # softmax is not invertible. we constrain the returned value to have mean zero.
    ret = jnp.log(y)
    ret -= ret.mean()
    return ret
