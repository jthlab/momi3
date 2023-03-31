import demes
import numpy as np

from momi3.MOMI import Momi

from .demos import ThreeDemes


def test_onepop():
    b = demes.Builder(description="demo")
    b.add_deme("A", epochs=[dict(start_size=1, end_time=10), dict(start_size=10)])
    demo = b.resolve()
    sampled_demes = ["A"]
    sample_sizes = [200]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=False)
    d = {"A": 10}
    s1 = momi.sfs_entry(d)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    momi_b = momi.bound(bounds)
    s2 = momi_b.sfs_entry(d)
    np.testing.assert_allclose(s1, s2)


def test_threepop_pulse():
    demo, _ = ThreeDemes.Constant().pulses()
    sampled_demes = ["A", "B", "C"]
    sample_sizes = 3 * [50]
    dict(zip(sampled_demes, sample_sizes))
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=False)
    d = dict(zip("ABC", range(10, 13)))
    s1 = momi.sfs_entry(d)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    momi_b = momi.bound(bounds)
    s2 = momi_b.sfs_entry(d)
    np.testing.assert_allclose(s1, s2)


def test_non_adm_non_mig():
    n = 30
    d = {"A": 1, "B": 1}
    demo = demes.load("tests/yaml_files/TwoDemes.yml")
    sampled_demes = tuple(["A", "B"])
    sample_sizes = (n, n)
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    s1 = momi.sfs_entry(d)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    momi_b = momi.bound(bounds)
    s2 = momi_b.sfs_entry(d)
    np.testing.assert_allclose(s1, s2)


def jacobson_bound_sampler():
    n = 4
    demo = demes.load("tests/yaml_files/jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
