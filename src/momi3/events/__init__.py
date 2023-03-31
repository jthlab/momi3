from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

from src.momi3.common import T

from ..common import Axes, PopCounter, Population, State
from .admix import Admix, Pulse
from .downsample import Downsample
from .event import Event
from .lift import Lift, MigrationStart
from .split1 import Split1
from .split2 import Split2

__all__ = [
    "NoOp",
    "Rename",
    "Split1",
    "Split2",
    "Admix",
    "Pulse",
    "Lift",
    "Downsample",
    "MigrationStart",
]


@dataclass(frozen=True, kw_only=True)
class Rename(Event):
    "Rename a population."
    old: Population
    new: Population

    def _setup_impl(
        self, in_axes: Axes, ns: Counter[Population, int]
    ) -> tuple[Axes, PopCounter, T]:
        out_axes = deepcopy(in_axes)
        out_axes[self.new] = out_axes.pop(self.old)
        ns = deepcopy(ns)
        ns[self.new] = ns.pop(self.old)
        return out_axes, ns, None

    def _execute_impl(self, st: State, params: dict, aux: T) -> State:
        return st


@dataclass(frozen=True, kw_only=True)
class NoOp(Event):
    def _setup_impl(self, in_axes: Axes, ns: PopCounter) -> tuple[Axes, PopCounter, T]:
        return in_axes, ns, None

    def _execute_impl(self, st: State, params: dict, aux: T) -> State:
        return st
