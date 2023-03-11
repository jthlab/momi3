import numpy as np
import demes

from momi3.event import ETBuilder
from momi3.MOMI import Momi
from momi3.Params import Params

from .demos import ThreeDemes


def test_pulse():
    demo, _ = ThreeDemes.Constant().pulses()

    sampled_demes = ["A", "B", "C"]
    sample_sizes = 3 * [50]
    num_samples = dict(zip(sampled_demes, sample_sizes))

    momi = Momi(demo, sampled_demes, sample_sizes, jitted=False)
    params = Params(momi)
    params.set_train("eta_0", True)
    params.set_train("eta_1", True)
    bounds = momi.bound_sampler(params, [0.0, 0.0], 100)
    ETBuilder(demo, num_samples, bounds)


def test_non_adm_non_mig():
    demo, _ = ThreeDemes.Exponential().base()

    sampled_demes = ["A", "B", "C"]
    sample_sizes = 3 * [50]

    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    d = {"A": 35, "B": 25, "C": 15}
    s1 = momi.sfs_entry(d)

    params = Params(momi)
    params.set_train("eta_0", True)
    params.set_train("eta_1", True)
    bounds = momi.bound_sampler(params, [0.0, 0.0], 100)
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
    s2 = momi.sfs_entry(d)

    np.testing.assert_allclose(s1, s2)


def jacobson_bound_sampler():
    n = 4
    demo = demes.load("yaml_files/jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
