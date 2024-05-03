from pathlib import Path

import demes
import numpy as np
from pytest import fixture


@fixture
def yaml_path():
    return Path(__file__).parent / "yaml_files"


@fixture
def rng():
    return np.random.default_rng(1)


@fixture
def iwm(yaml_path) -> "demes.Graph":
    return demes.load(yaml_path / "IWM.yml")
