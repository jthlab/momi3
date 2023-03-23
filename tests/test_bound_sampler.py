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

    n = 30
    d = {'A': 1, 'B': 1}

    demo = demes.load("yaml_files/TwoDemes.yml")
    sampled_demes = tuple(["A", "B"])
    sample_sizes = (n, n)
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    s1 = momi.sfs_entry(d)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    momi_b = Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
    s2 = momi_b.sfs_entry(d)
    np.testing.assert_allclose(s1, s2)


def jacobson_bound_sampler():
    n = 4
    demo = demes.load("yaml_files/jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
