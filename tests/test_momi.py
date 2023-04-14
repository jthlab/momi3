import numpy as np
import moments
import demes
import pytest

# from momi3 import esfs
from momi3.MOMI import Momi

#from .demos import TwoDemes, ThreeDemes

from momi3.utils import Parallel_runtime, update
from scipy.optimize import approx_fprime


def test_esfs(iwm):
    sample_sizes = {"deme0": 5, "deme1": 3}
    e = esfs(iwm, sample_sizes)
    assert e.shape == (6, 4)


@pytest.mark.rate0
def test_two_pop_migration_0():
    t = 10
    size = 100
    demo, _ = TwoDemes.Constant(t=t, size=size).base()
    demo_m, _ = TwoDemes.Constant(t=t, size=size).migration_sym(t, 0, rate=0.0)
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 6]
    spec_demo = Momi(demo, sampled_demes, sample_sizes).sfs_spectrum()
    spec_demo_m = Momi(demo_m, sampled_demes, sample_sizes).sfs_spectrum()
    assert np.allclose(spec_demo_m, spec_demo, rtol=1e-4), np.nanmean(
        np.abs(spec_demo_m - spec_demo) / spec_demo
    )


@pytest.mark.rate0
def test_two_pop_migration_exp_growth_0():
    t = 800
    g = 0.005
    size = 10000
    demo, _ = TwoDemes.Exponential(t=t, g=g, size=size).base()
    demo_m, _ = TwoDemes.Exponential(t=t, g=g, size=size).migration_sym(t, 0, rate=0.0)
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 6]
    spec_demo = Momi(demo, sampled_demes, sample_sizes).sfs_spectrum()
    spec_demo_m = Momi(demo_m, sampled_demes, sample_sizes).sfs_spectrum()
    assert np.allclose(spec_demo_m, spec_demo, rtol=1e-4), np.nanmean(
        np.abs(spec_demo_m - spec_demo) / spec_demo
    )


@pytest.mark.rate0
def test_three_pop_migration_exp_growth_0():
    t = 800
    g = 0.005
    size = 10000
    demo, _ = ThreeDemes.Exponential(t=t, g=g, size=size).base()
    demo_m, _ = ThreeDemes.Exponential(t=t, g=g, size=size).migrations(rate=0.0)
    sampled_demes = ["A", "B", "C"]
    sample_sizes = [4, 6, 3]
    spec_demo = Momi(demo, sampled_demes, sample_sizes).sfs_spectrum()
    spec_demo_m = Momi(demo_m, sampled_demes, sample_sizes).sfs_spectrum()
    l1 = lambda x, y: np.abs(x - y).mean()
    print(l1(spec_demo, spec_demo_m))
    assert np.allclose(spec_demo_m, spec_demo, rtol=1e-4), np.nanmean(
        np.abs(spec_demo_m - spec_demo) / spec_demo
    )


def test_grad_speed_momi_moments_gutenkunst():
    def moments_loglik_with_gradient_time(params, sampled_demes, sample_sizes, jsfs):
        jsfs_flatten = jsfs.todense().flatten()
        demo_dict = params.demo_dict
        theta_train = np.array(params._theta_train)
        keys = list(params._theta_train_dict)

        def loglik(theta_train, jsfs_flatten):
            theta_train_dict = dict(zip(keys, theta_train))

            for paths, val in theta_train_dict.items():
                for path in paths:
                    update(demo_dict, path, float(val))

            demo = demes.Builder.fromdict(demo_dict).resolve()
            esfs = moments.Spectrum.from_demes(
                demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
            )
            esfs = np.array(esfs).flatten()[1:-1]
            data = jsfs_flatten[1:-1]
            esfs /= esfs.sum()
            esfs = np.clip(1e-32, np.inf, esfs)
            return (data * np.log(esfs)).sum()

        def grad(theta_train, jsfs_flatten):
            return approx_fprime(
                theta_train, loglik, 0.1, jsfs_flatten
            )

        def fwg():
            v = loglik(theta_train, jsfs_flatten)
            g = grad(theta_train, jsfs_flatten)
            return v, g

        return Parallel_runtime(fwg, num_replicates=4, n_jobs=4)

    n = 10
    nmut = 5000
    BATCH_SIZE = 1000

    demo = demes.load("gutenkunst_data/gutenkunst_ooa.yml")
    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = 3 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    jsfs = momi.simulate(nmut, seed=108)
    print(f"non-zero-entries: {jsfs.nnz}")

    params = momi._default_params
    params.set_train_all_rhos(True)
    params.set_train_all_etas(True)

    (v, g), compilation_time, runtime = momi._time_loglik_with_gradient(
        params, jsfs, repeat=5, batch_size=BATCH_SIZE
    )

    moments_times = moments_loglik_with_gradient_time(params, sampled_demes, sample_sizes, jsfs)
    moments_times = np.median(moments_times)

    print(f"Single grad iter by moments takes {moments_times:.2f} secs")
    print(f"Single grad iter by momi3 takes {runtime:.2f} secs")
    print(f"Compilation takes {compilation_time:.2f} secs")


def test_pop_shrink_w_mig():
    demo = demes.load('yaml_files/3_pop_mig.yaml')
    sampled_demes = ['YRI', 'CEU', 'CHB']
    sample_sizes = 3 * [20]

    new_vals = {
        'log(eta_1)': 9.421890194466682,
        'log(eta_2)': 6.759644478755238,
        'log(eta_3)': 11.588151254546482,
        'log(eta_4)': 10.913100751770104,
        'log(eta_5)': 12.011600865606297,
        'log(eta_6)': 12.362115232659324,
        'logit(rho_0)': -2.260751721723769,
        'logit(rho_1)': -8.455725492291322,
        'logit(rho_2)': -6.074292717977304
    }

    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    params = momi._default_params
    params.set_train_all_rhos(True)
    params.set_train_all_etas(True)
    params.set_train('eta_0', False)
    params.set_optimization_results(new_vals)

    demo = params.demo_graph
    momi_m = Momi(demo, sampled_demes, sample_sizes, jitted=True)

    ddict = params.demo_dict
    ddict['migrations'] = []
    demo_non_mig = demes.Builder.fromdict(ddict).resolve()
    momi_v = Momi(demo_non_mig, sampled_demes, sample_sizes, jitted=True)

    jsfs = momi_v.simulate(100, seed=108)

    for momi in momi_v, momi_m:
        params = momi._default_params
        params.set_train_all_etas(True)

        x = momi.loglik_with_gradient(params, jsfs)
        print('loglik:', x[0])
        print('grad:')
        for i in x[1]:
            print(i, x[1][i])


if __name__ == '__main__':
    test_pop_shrink_w_mig()
