"""Tests for Expected Normalized SFS spectrums
Assesses 100 * np.abs(sfs_{momi3} - sfs_{momi2}) / sfs_{momi2} <= MAX_AVG_PERCENT_ERROR
Assesses 100 * np.abs(sfs_{momi3} - sfs_{moments}) / sfs_{moments} <= MAX_AVG_PERCENT_ERROR
Sfs' are normalized
For testing: pytest -W ignore tests/tests_moments_momi2.py
For debugging/printing sfs entries: python tests/tests_moments_momi2.py <run_type> <test> NORMALIZE_ESFS=<bool> order_by_error=<bool> no_entries=<int>
"""  # noqa: E501
import sys
from copy import deepcopy
from itertools import product

import dadi
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
dadi_pts = 200

jax.config.update("jax_enable_x64", False)


class Momi_vs_Moments:
    def __init__(self, demo, model1, sampled_demes, sample_sizes):
        self.sampled_demes = sampled_demes
        self.sample_sizes = sample_sizes
        self.momi_graph = demo
        self.moments_graph = demo
        self.momi2_model = model1

    @cached_property
    def momi_sfs(self):
        dG = self.momi_graph
        momi = Momi(
            dG, sampled_demes=self.sampled_demes, sample_sizes=self.sample_sizes
        )
        esfs = momi.sfs_spectrum()
        return esfs

    @cached_property
    def moments_sfs(self):
        dG = self.moments_graph

        esfs = moments.Spectrum.from_demes(
            dG, sampled_demes=self.sampled_demes, sample_sizes=self.sample_sizes
        )
        esfs = np.array(esfs)
        return esfs * 4 * dG.demes[0].epochs[0].start_size

    @cached_property
    def dadi_sfs(self):
        dG = self.moments_graph

        esfs = dadi.Spectrum.from_demes(
            dG,
            sampled_demes=self.sampled_demes,
            sample_sizes=self.sample_sizes,
            pts=dadi_pts,
        )
        esfs = np.array(esfs)
        return esfs * 4 * dG.demes[0].epochs[0].start_size

    @cached_property
    def momi2_sfs(self):
        momi2_model = self.momi2_model
        if momi2_model is None:
            raise ValueError("momi2 model is not defined")
        P = len(self.sample_sizes)
        momi2sfs_array = []
        Mutant_sizes = np.array(
            list(product(*[range(n + 1) for n in self.sample_sizes]))
        )[1:-1]
        for bs in Mutant_sizes:
            x = []
            for i in range(P):
                n = self.sample_sizes[i]
                b = bs[i]
                x.append([int(n - b), int(b)])
            momi2sfs_array.append([x])
        s = momi2.site_freq_spectrum(momi2_model.leafs, momi2sfs_array)
        momi2_model.set_data(s, length=1)
        momi2_model.set_mut_rate(1.0)
        ret = momi2_model.expected_sfs()
        esfs = np.zeros([n + 1 for n in self.sample_sizes])
        for key in ret:
            ind = [i[1] for i in key]
            esfs[tuple(ind)] = ret[key]
        return esfs

    def get_flatten_spectrum(self, method, normalize=NORMALIZE_ESFS):
        if method == "momi3":
            sfs = self.momi_sfs.flatten()[1:-1]
        elif method == "moments":
            sfs = self.moments_sfs.flatten()[1:-1]
        elif method == "dadi":
            sfs = self.dadi_sfs.flatten()[1:-1]
        elif method == "momi2":
            sfs = self.momi2_sfs.flatten()[1:-1]
        else:
            raise ValueError(f"Unknown test:{method}")

        if normalize:
            return sfs / sfs.sum()
        else:
            return sfs

    def compare(self, method1, method2, run_type, **kwargs):
        if "NORMALIZE_ESFS" not in kwargs:
            kwargs["NORMALIZE_ESFS"] = NORMALIZE_ESFS

        m1 = self.get_flatten_spectrum(method1, kwargs["NORMALIZE_ESFS"])
        m2 = self.get_flatten_spectrum(method2, kwargs["NORMALIZE_ESFS"])
        ape = absolute_percent_error(m1, m2)

        if run_type == "pytest":
            mape = np.mean(ape)
            assert (
                mape <= MAX_AVG_PERCENT_ERROR
            ), f"MAPE({method1}, {method2}) = {mape:0.2f}% > {MAX_AVG_PERCENT_ERROR}%"

        elif run_type == "mape":
            mape = np.mean(ape)
            print(f"MAPE({method1}, {method2}) = {mape:0.2f}%")

        elif run_type[0] == "l":
            p = int(run_type[1:])
            x = lnorm(m1, m2, p)
            x = x.sum()
            print(f"{run_type}({method1}, {method2}) = {x:0.2f}")

        elif run_type == "debug":
            if "order_by_error" not in kwargs:
                kwargs["order_by_error"] = False
            if "no_entries" not in kwargs:
                kwargs["no_entries"] = len(m1)

            Mutant_sizes = np.array(
                list(product(*[range(n + 1) for n in self.sample_sizes]))
            )[1:-1]

            if kwargs["order_by_error"]:
                sorted_index = np.argsort(-ape)
            else:
                sorted_index = np.arange(len(ape))

            ape = iter(ape[sorted_index])

            sfs1 = iter(m1[sorted_index])
            sfs2 = iter(m2[sorted_index])

            Mutant_sizes = iter(Mutant_sizes[sorted_index])

            print(
                "{:<25}\t{:>10}\t{:>10}\t{:>10}".format(
                    "deriveds", "error", method1, method2
                )
            )

            for _ in range(kwargs["no_entries"]):
                print(
                    "{mut:<25}\t{val:>10.2g}\t{sfs1:>10.3g}\t{sfs2:>10.3g}".format(
                        mut=str(next(Mutant_sizes)),
                        val=next(ape),
                        sfs1=next(sfs1),
                        sfs2=next(sfs2),
                    )
                )
        else:
            raise ValueError(f"Unknown run_type {run_type}")

    def multinomial_log_likelihood(self, method, jsfs):
        esfs = self.get_flatten_spectrum(method, normalize=True)
        sfs = jsfs.todense().flatten()[1:-1]
        return multinomial_log_likelihood(sfs, esfs, esfs.sum())

    def compare_loglik(self, method1, method2, jsfs, run_type):
        loglik_method1 = self.multinomial_log_likelihood(method1, jsfs)
        loglik_method2 = self.multinomial_log_likelihood(method2, jsfs)

        if run_type == "pytest":
            # assert loglik(method_1, jsfs) > loglik(method_2, jsfs) - 2
            assert loglik_method1 > loglik_method2 - 2
        elif run_type == "debug":
            print(f"loglik {method1} = {loglik_method1:.2f}")
            print(f"loglik {method2} = {loglik_method2:.2f}")
        else:
            print("Cannot handle other runtypes")


def lnorm(vec1, vec2, p=1):
    x = np.abs(vec1 - vec2)
    return np.power(x, p)


def absolute_percent_error(vec1, vec2, normalize=NORMALIZE_ESFS):
    return 100 * np.abs(vec1 - vec2) / vec2


def generate_fwdpy11_test(model, run_type="pytest", **kwargs):
    jsfs = sparse.load_npz(f"tests/out_files/{model}.npz")
    demo = demes.load(f"tests/yaml_files/{model}.yml")
    mvm = Momi_vs_Moments(
        demo, None, demo.metadata["sampled_demes"], demo.metadata["sample_sizes"]
    )
    mvm.compare_loglik("momi3", "moments", jsfs, run_type)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def generate_grid_test(
    demo, sampled_demes, sample_sizes, params_to_change, grid_values, run_type="pytest"
):
    assert len(grid_values) == len(params_to_change)
    assert np.var([len(x) for x in grid_values]) == 0

    def get_moments_esfs(demo):
        return np.array(
            moments.Spectrum.from_demes(
                demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
            )
        ).flatten()[1:-1]

    momi = Momi(
        demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes, jitted=True
    )

    params = Params(Momi(demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes))

    n_grid = len(grid_values[0])
    p = len(params_to_change)

    for i in range(n_grid):
        for j in range(p):
            key = params_to_change[j]
            val = grid_values[j][i]
            params.set(key, val)

        cur_demo = params.demo_graph

        moments_esfs = get_moments_esfs(cur_demo)

        momi3_esfs = momi.sfs_spectrum(params).flatten()[1:-1]

        ape = absolute_percent_error(momi3_esfs, moments_esfs, True)
        mape = np.mean(ape)

        cur_params = dict(zip(params_to_change, [grid_values[j][i] for j in range(p)]))
        message = [f"MAPE(momi3, moments) = {mape:0.2f}%", f" for {cur_params}"]
        if run_type == "pytest":
            message = message[0] + f" > {MAX_AVG_PERCENT_ERROR}%" + message[1]
            assert mape <= MAX_AVG_PERCENT_ERROR, message
        elif run_type == "mape":
            print(message[0] + message[1])
        elif run_type == "debug":
            raise ValueError("debugging version has not written yet")
        else:
            raise ValueError(f"Unknown run_type {run_type}")


def compare_gradients(momi, params, sampled_demes, sample_sizes):
    theta_train, theta_nuisance = params._loglik_arrays
    G_ts = params._loglik_G_ts
    data = momi.data
    demo_dict = params.demo_dict

    def loglik_moments_with_params(
        theta_train,
        theta_nuisance=theta_nuisance,
        G_ts=G_ts,
        data=data,
        demo_dict=demo_dict,
        sampled_demes=sampled_demes,
        sample_sizes=sample_sizes,
    ):
        theta = jnp.concatenate([theta_train, theta_nuisance])
        theta = [float(i) for i in theta]
        demo_dict = deepcopy(demo_dict)
        for vec_i, g_ts in zip(theta, G_ts):
            for g_t in g_ts:
                demo_dict = g_t(demo_dict, vec_i)
        demo = demes.Builder.fromdict(demo_dict).resolve()
        esfs = moments.Spectrum.from_demes(
            demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
        )
        esfs = np.array(esfs).flatten()[1:-1]
        jsfs = data._jsfs().todense().flatten()[1:-1]
        return multinomial_log_likelihood(jsfs, esfs, esfs.sum())

    _, a_grad = momi.loglik(params, with_grad=True)

    def f(epsilon):
        n_grad = approx_fprime(theta_train, loglik_moments_with_params, epsilon=epsilon)
        return np.square(a_grad - n_grad).sum()

    eps_hat = scipy.optimize.minimize_scalar(f, bounds=(1e-8, 0.1), method="bounded").x
    n_grad = approx_fprime(theta_train, loglik_moments_with_params, epsilon=eps_hat)

    assertion_message = (
        f"momi3_grad = {[float(i) for i in a_grad]}, numerical_grad_moments = {n_grad} "
    )
    assert np.allclose(a_grad, n_grad, atol=0, rtol=GRADIENT_RTOL), assertion_message


# SINGLE DEME TESTS


def test_single_pop(run_type="pytest", **kwargs):
    size = 10.0

    print("single pop")
    demo, model1 = SingleDeme.Constant(size).base()
    sampled_demes = ["A"]
    sample_sizes = [10]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_single_pop_small_exponential_size(run_type="pytest", **kwargs):
    g = 1e-5

    demo, model1 = SingleDeme.Exponential(g=g).base()
    sampled_demes = ["A"]
    sample_sizes = [10]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"single pop exponential growth w/ {g=}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_single_pop_exponential_size(run_type="pytest", **kwargs):
    g = 10.0
    size = 100.0

    demo, model1 = SingleDeme.Exponential(size=size, g=g).base()
    sampled_demes = ["A"]
    sample_sizes = [10]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"single pop exponential growth w/ {g=} and end_size {size=}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_single_pop_exponential_size_minus(run_type="pytest", **kwargs):
    g = -1.5

    demo, model1 = SingleDeme.Exponential(g=g).base()
    sampled_demes = ["A"]
    sample_sizes = [10]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"single pop exponential growth w/ {g=}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


# TWO DEMES TESTS


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_two_pop_n1234(n, run_type="pytest", **kwargs):
    size = 1.0
    demo, model1 = TwoDemes.Constant(size=size).base()
    sampled_demes = ["A", "B"]
    sample_sizes = [n, n + 1]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop(run_type="pytest", **kwargs):
    size = 1.0
    demo, model1 = TwoDemes.Constant(size=size).base()
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_pulse_0(run_type="pytest", **kwargs):
    size = 1.0
    p = 0.0

    demo, model1 = TwoDemes.Constant(size=size).pulse(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop with pulse 0")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_two_pulses_0(run_type="pytest", **kwargs):
    size = 1.0
    p1 = p2 = 0.0
    tp1, tp2 = 0.25, 0.75

    demo, model1 = TwoDemes.Constant(size=size).two_pulses(
        tp1=tp1, tp2=tp2, p1=p1, p2=p2
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop w/ two pulses each 0")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_pulse_09(run_type="pytest", **kwargs):
    size = 1.0
    p = 0.9

    demo, model1 = TwoDemes.Constant(size=size).pulse(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop w/ pulse = {p}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    # mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_pulse_05(run_type="pytest", **kwargs):
    size = 1.0
    p = 0.5

    demo, model1 = TwoDemes.Constant(size=size).pulse(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop w/ pulse = {p}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_pulse_01(run_type="pytest", **kwargs):
    size = 1.0
    p = 0.1

    demo, model1 = TwoDemes.Constant(size=size).pulse(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop w/ pulse = {p}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_two_pulses(run_type="pytest", **kwargs):
    size = 1.0
    p1 = p2 = 0.1
    tp1, tp2 = 0.25, 0.75

    demo, model1 = TwoDemes.Constant(size=size).two_pulses(
        tp1=tp1, tp2=tp2, p1=p1, p2=p2
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop w/ two pulses")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_two_pulses_large_n(run_type="pytest", **kwargs):
    demo, model1 = TwoDemes.Constant().two_pulses()
    sampled_demes = ["A", "B"]
    sample_sizes = [30, 31]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    mvm.compare("momi3", "momi2", run_type, **kwargs)


def test_two_pop_pulse_sym(run_type="pytest", **kwargs):
    size = 1.0
    p1 = p2 = 0.1
    tp1 = tp2 = 0.5

    demo, model1 = TwoDemes.Constant(size=size).two_pulses(
        tp1=tp1, tp2=tp2, p1=p1, p2=p2
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop w/ two pulses each 0")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_migration_two_phase(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1
    tstart = t
    tend = 0.0
    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_twophase(
        rate=rate, tstart=tstart, tend=tend
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop IWM between [0, {t/2}], [{t/2}, {t}]")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_interrupted_migration1(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1
    tstart = t / 2
    tend = 0.0

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration(
        rate=rate, tstart=tstart, tend=tend
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop IWM between [0, {t/2}]")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_interrupted_migration2(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1
    tstart = t
    tend = t / 2

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration(
        rate=rate, tstart=tstart, tend=tend
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop IWM between [{t/2}, 1]")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_IWM(run_type="pytest", **kwargs):
    size = 10.0
    t = 1.0
    rate = 0.1
    demo, model1 = TwoDemes.Constant(size=size, t=t).migration(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop IWM")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_two_pop_IWM_small_n(n, run_type="pytest", **kwargs):
    size = 10.0
    t = 1.0
    rate = 0.1
    demo, model1 = TwoDemes.Constant(size=size, t=t).migration(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [n, n + 1]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop IWM")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_IWM_sym(run_type="pytest", **kwargs):
    size = 1.0
    t = 0.5
    rate = 0.1

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop IWM with symmetric rate")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_two_pop_IWM_sym_pulse_0(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1
    p = 0

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym_pulse(
        p=p, rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop IWM with pulse = {p}")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.parametrize(
    "size,t", [(1.0, 1.0), (10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (100.0, 1.0)]
)
def test_two_pop_IWM_sym_pulse(size, t, run_type="pytest", **kwargs):
    rate = 0.1
    p = 0.1
    tp = t / 2

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym_pulse(
        tp=tp, p=p, rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 8]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print(f"two-pop IWM with pulse and migration with {p=} and {rate=}")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_two_pop_exponential_minus(run_type="pytest", **kwargs):
    g = -1.0
    size = 50.0

    demo, model1 = TwoDemes.Exponential(size=size, g=g).base()
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop exp pop shrink")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_two_pop_exponential(run_type="pytest", **kwargs):
    g = 5.0
    size = 50.0

    demo, model1 = TwoDemes.Exponential(size=size, g=g).base()
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop exp pop growth")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_two_pop_exponential_pulse_0(run_type="pytest", **kwargs):
    g = 5.0
    size = 10.0
    p = 0.0

    demo, model1 = TwoDemes.Exponential(size=size, g=g).pulse(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop exp growth w/ pulse 0")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_two_pop_exponential_two_pulses_0(run_type="pytest", **kwargs):
    g = 5.0
    size = 10.0
    p = 0.0

    demo, model1 = TwoDemes.Exponential(size=size, g=g).two_pulses(p1=p, p2=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop exp growth w/ two pulses each 0")
    mvm.compare("momi3", "momi2", run_type, **kwargs)


def test_two_pop_five_pulses(run_type="pytest", **kwargs):
    size = 1.0
    p = 0.0

    demo, model1 = TwoDemes.Constant(size=size).five_pulses(p=p)
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 3]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop w/ five pulses = {p}")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.migration
@pytest.mark.exponential
def test_two_pop_exponential_migration(run_type="pytest", **kwargs):
    g = 0.025
    size = 1000.0
    rate = 0.01
    t = 100.0

    demo, model1 = TwoDemes.Exponential(t=t, size=size, g=g).migration(
        tstart=t, rate=rate
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("two-pop exp growth w/ migration")
    # mvm.compare("moments", "dadi", run_type, **kwargs)
    # mvm.compare("momi3", "dadi", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


# MORE THAN 2


def test_three_pop(run_type="pytest", **kwargs):
    size = 1.0
    t1 = 1.0
    t2 = 1.01

    demo, model1 = ThreeDemes.Constant(size=size, t1=t1, t2=t2).base()
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [5, 5, 5]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.exponential
def test_three_pop_exponential(run_type="pytest", **kwargs):
    size = 1.0

    demo, model1 = ThreeDemes.Exponential(size=size).base()
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [6, 5, 4]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop exponential growth")
    mvm.compare("momi3", "momi2", run_type, **kwargs)
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_three_pop_two_pulses(run_type="pytest", **kwargs):
    size = 1.0
    npulses = 2

    demo, model1 = ThreeDemes.Constant(size=size).pulses(npulses=npulses)
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [6, 5, 4]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop {npulses} pulses")
    mvm.compare("momi3", "momi2", run_type, **kwargs)


@pytest.mark.migration
def test_three_pop_migrations(run_type="pytest", **kwargs):
    size = 1.0
    rate = 0.01
    t1 = 1.0
    t2 = 1.2

    demo, model1 = ThreeDemes.Constant(size=size, t1=t1, t2=t2).three_migrants(
        rate=rate
    )
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [5, 5, 5]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop multiple migrations")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.migration
@pytest.mark.exponential
def test_three_pop_exponential_migrations(run_type="pytest", **kwargs):
    size = 1.0
    rate = 0.01

    demo, model1 = ThreeDemes.Exponential(size=size).migrations(rate=rate)
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [6, 5, 4]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop exponential growth with migration")
    mvm.compare("momi3", "moments", run_type, **kwargs)


@pytest.mark.migration
@pytest.mark.exponential
def test_three_pop_exponential_pulse_migration(run_type="pytest", **kwargs):
    t = 10
    g = 0.25
    tp1 = 5
    tp2 = 7

    demo, model1 = ThreeDemes.Exponential(t=t, g=g).pulses_migration(tp1=tp1, tp2=tp2)
    sampled_demes = ["A", "B", "C"]
    sample_sizes = 3 * [5]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop exponential growth with migration")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_three_pop_multi_anc(run_type="pytest", **kwargs):
    demo, model1 = MultiAnc().base()
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [10, 8, 5]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("three-pop multi-anc")
    mvm.compare("momi3", "moments", run_type, **kwargs)


def test_five_pops(run_type="pytest", **kwargs):
    demo, model1 = FiveDemes().base()
    sampled_demes = ["A", "B", "C", "D", "E"]
    sample_sizes = [4, 3, 5, 5, 3]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("five-pop")
    mvm.compare("momi3", "moments", run_type, **kwargs)
    mvm.compare("momi3", "momi2", run_type, **kwargs)


def test_five_pops_pulses(run_type="pytest", **kwargs):
    demo, model1 = FiveDemes().pulses()
    sampled_demes = ["A", "B", "C", "D", "E"]
    sample_sizes = [4, 3, 5, 5, 3]
    mvm = Momi_vs_Moments(demo, model1, sampled_demes, sample_sizes)
    print("five-pop")
    mvm.compare("momi3", "moments", run_type, **kwargs)
    mvm.compare("momi3", "momi2", run_type, **kwargs)


def test_gutenkunst(run_type="pytest", **kwargs):
    demo = demes.load("tests/yaml_files/gutenkunst_ooa.yml")
    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = [4, 4, 4]
    mvm = Momi_vs_Moments(demo, None, sampled_demes, sample_sizes)
    mvm.compare("momi3", "moments", run_type, **kwargs)


# FWDPY11 tests
# These tests are comparing log-likelihood values of momi3 and moments
# They are `not` running a fwdpy11 simulation. They assume the simualtion joint-sfs is located at:
# jsfs = sparse.load_npz(f"tests/out_files/{model}.npz")
# And associated demography at:
# demo = demes.load(f"tests/yaml_files/{model}.yml")


def test_fwdpy11_IWM(run_type="pytest", **kwargs):
    model = "IWM"
    generate_fwdpy11_test(model, run_type, **kwargs)


def test_fwdpy11_IWMA(run_type="pytest", **kwargs):
    model = "IWMA"
    generate_fwdpy11_test(model, run_type, **kwargs)


# Grid tests of momi3 vs moments
# Comparing esfs of momi3 with moments for grids of given parameters
# grid_values of params_to_change[i] are grid_values[i]


@pytest.mark.migration
def test_two_pop_IWM_grid_same_sizes(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]

    params_to_change = ["eta_0", "eta_1", "eta_2"]
    grid_values = 3 * [[0.2, 1.0, 10.0, 15.0, 100.0, 1000.0]]

    generate_grid_test(
        demo, sampled_demes, sample_sizes, params_to_change, grid_values, run_type
    )


@pytest.mark.migration
def test_two_pop_IWM_grid_split_time_size(run_type="pytest", **kwargs):
    size = 1.0
    t = 1.0
    rate = 0.1

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]

    params_to_change = ["tau_1", "eta_0", "eta_1", "eta_2"]

    tau_vals = [0.2, 1.0, 10.0, 100.0]
    size_vals = [0.2, 1.0, 10.0, 15.0, 100.0, 1000.0]

    grid_values_t = []
    grid_values_s = []

    for t, s in product(tau_vals, size_vals):
        grid_values_t.append(t)
        grid_values_s.append(s)

    grid_values = [grid_values_t] + 3 * [grid_values_s]

    generate_grid_test(
        demo, sampled_demes, sample_sizes, params_to_change, grid_values, run_type
    )


@pytest.mark.migration
def test_two_pop_IWM_grid_rate(run_type="pytest", **kwargs):
    size = 1.0
    t = 2.0
    rate = 0.1

    demo, model1 = TwoDemes.Constant(size=size, t=t).migration_sym(
        rate=rate, tstart=t, tend=0
    )
    sampled_demes = ["A", "B"]
    sample_sizes = [10, 6]

    params_to_change = ["rho_0", "rho_1"]
    grid_values = 2 * [[0.01, 0.1, 0.2]]

    generate_grid_test(
        demo, sampled_demes, sample_sizes, params_to_change, grid_values, run_type
    )


if __name__ == "__main__":
    # print(f"Device: {jax.devices()}\n")

    run_type = sys.argv[1]
    test = sys.argv[2]
    kwargs = ", ".join(sys.argv[3:])

    if run_type in ["pytest", "mape", "debug"]:
        exec(f"{test}(run_type='{run_type}', {kwargs})")
    else:
        raise ValueError(f"Unknown run_type={sys.argv[1]}")
