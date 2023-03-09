import numpy as np
from pytest import fixture


@fixture
def rng():
    return np.random.default_rng(1)
