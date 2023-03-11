import numpy as np

from momi3 import esfs
from momi3.MOMI import Momi

from .demos import TwoDemes


def test_esfs(iwm):
    sample_sizes = {"deme0": 5, "deme1": 3}
    e = esfs(iwm, sample_sizes)
    assert e.shape == (6, 4)


def test_two_pop_migration_exp_growth_0():
    t = 10
    g = 0.1
    size = 100
    demo, _ = TwoDemes.Exponential(t=t, g=g, size=size).base()
    demo_m, _ = TwoDemes.Exponential(t=t, g=g, size=size).migration_sym(t, 0, rate=0.0)
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 6]
    spec_demo = Momi(demo, sampled_demes, sample_sizes).sfs_spectrum()
    spec_demo_m = Momi(demo_m, sampled_demes, sample_sizes).sfs_spectrum()
    assert np.allclose(spec_demo_m, spec_demo, rtol=1e-4), np.nanmean(
        np.abs(spec_demo_m - spec_demo) / spec_demo
    )
