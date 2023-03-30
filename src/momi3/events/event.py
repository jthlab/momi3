from collections import Counter
from dataclasses import dataclass

from src.momi3.common import Axes, PopCounter, Population, State, T


@dataclass(frozen=True, kw_only=True)
class Event:
    "Base class for events."
    downsample: Axes = None

    def setup(
        self, axes: Axes, ns: Counter[Population, int]
    ) -> tuple[Axes, PopCounter, T]:
        return self._setup_impl(axes, ns)

    def execute(self, st: State, params: dict, aux: T) -> State:
        return self._execute_impl(st, params, aux)
