import sys
from copy import deepcopy
from itertools import product

import demes
import jax
import jax.numpy as jnp
import moments
import momi as momi2
import numpy as np
import pytest
import scipy
import sparse
from cached_property import cached_property
from scipy.optimize import approx_fprime

from momi3.JAX_functions import multinomial_log_likelihood
from momi3.MOMI import Momi
from momi3.Params import Params
from tests.demos import FiveDemes, MultiAnc, SingleDeme, ThreeDemes, TwoDemes

MAX_AVG_PERCENT_ERROR = 1.0
GRADIENT_RTOL = 0.05
NORMALIZE_ESFS = True


def test_two_pop_migration_exp_growth_0():
	t = 10
	g = 0.1
	size = 100

	demo, _ = TwoDemes.Exponential(t=t, g=g, size=size).base()
	demo_m, _ = TwoDemes.Exponential(t=t, g=g, size=size).migration_sym(t, 0, rate=0.)

	sampled_demes = ["A", "B"]
	sample_sizes = [4, 6]
	spec_demo = Momi(demo, sampled_demes, sample_sizes).sfs_spectrum()
	spec_demo_m = Momi(demo_m, sampled_demes, sample_sizes).sfs_spectrum()
	assert np.allclose(spec_demo_m, spec_demo), np.nanmean(np.abs(spec_demo_m - spec_demo) / spec_demo)
