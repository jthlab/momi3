import demes
import numpy as np
from pytest import fixture


@fixture
def rng():
    return np.random.default_rng(1)


@fixture
def iwm() -> "demes.Graph":
    return demes.load("tests/yaml_files/IWM.yml")
