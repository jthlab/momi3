import timeit
from copy import deepcopy

import autograd.numpy as np
import demes
import jax.numpy as jnp
import moments
import momi as momi2
import scipy
import sparse
from autograd import grad as auto_grad
from scipy.optimize import approx_fprime
from sparse._coo.core import COO

from momi3.JAX_functions import multinomial_log_likelihood
from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import Parallel_runtime, update
from tests.demos import SingleDeme, TwoDemes

GRADIENT_RTOL = 0.05


class momi2_timeit:
    def __init__(self, params, model, sampled_demes, sample_sizes):
        PD = dict(zip(params._nuisance_keys, np.array(params._theta_nuisance)))
        train_keys = params._train_keys

        def momi2_model(theta_train):
            return model(theta_train, train_keys=train_keys, PD=PD)

        num_sample = dict(zip(sampled_demes, sample_sizes))
        self.train_keys = train_keys
        self.sampled_demes = sampled_demes
        self.theta_train = np.array(params._theta_train)

        def momi2_model_func(x):
            return momi2_model(x)._get_demo(num_sample)

        def loglik(theta_train, sfs):
            return momi2.likelihood._composite_log_likelihood(
                sfs, momi2_model_func(theta_train)
            )

        self._loglik = loglik

        def grad(theta_train, sfs):
            return auto_grad(
                lambda x: momi2.likelihood._composite_log_likelihood(
                    sfs, momi2_model_func(x)
                )
            )(theta_train)

        self._grad = grad

    def loglik(self, jsfs):
        sfs = self.get_data(jsfs)
        return self._loglik(self.theta_train, sfs)

    def grad(self, jsfs):
        sfs = self.get_data(jsfs)
        g = self._grad(self.theta_train, sfs)
        return dict(zip(self.train_keys, [float(i) for i in g]))

    def time_loglik(self, jsfs, num_replicates, n_jobs):
        sfs = self.get_data(jsfs)

        def f():
            return self._loglik(self.theta_train, sfs)

        return Parallel_runtime(f, num_replicates, n_jobs)

    def time_grad(self, jsfs, num_replicates, n_jobs):
        sfs = self.get_data(jsfs)

        def f():
            return self._grad(self.theta_train, sfs)

        return Parallel_runtime(f, num_replicates, n_jobs)

    def get_data(self, jsfs):
        data = np.array(jsfs.data)
        coords = np.array(jsfs.coords.T)
        n = [i - 1 for i in jsfs.shape]
        P = len(n)
        config_list = {
            tuple((n[j] - coord[j], coord[j]) for j in range(P)): val
            for coord, val in zip(coords, data)
        }
        sfs = momi2.site_freq_spectrum(self.sampled_demes, [config_list])
        return sfs


class moments_timeit:
    def __init__(self, params, sampled_demes, sample_sizes):
        demo_dict = params.demo_dict
        self.theta_train = np.array(params._theta_train)
        self.EPS = 0.1

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

        self._loglik = loglik

        def grad(theta_train, jsfs_flatten):
            return approx_fprime(theta_train, loglik, self.EPS, jsfs_flatten)

        self._grad = grad

    def loglik(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._loglik(self.theta_train, jsfs_flatten)

    def grad(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._grad(self.theta_train, jsfs_flatten)

    def time_loglik(self, jsfs, num_replicates, n_jobs):
        jsfs_flatten = self.flatten_jsfs(jsfs)

        def f():
            return self._loglik(self.theta_train, jsfs_flatten)

        return Parallel_runtime(f, num_replicates, n_jobs)

    def time_grad(self, jsfs, num_replicates, n_jobs):
        jsfs_flatten = self.flatten_jsfs(jsfs)

        def f():
            return self._grad(self.theta_train, jsfs_flatten)

        return Parallel_runtime(f, num_replicates, n_jobs)

    def flatten_jsfs(self, jsfs):
        return jsfs.todense().flatten()


class momi_timeit:
    def __init__(self, params, sampled_demes, sample_sizes, batch_size=None):
        demo = params.demo_graph
        self.params = params
        self.momi = Momi(
            demo, sampled_demes, sample_sizes, jitted=True, batch_size=batch_size
        )

    def time(self, f, num_replicates):
        return timeit.repeat(f, number=1, repeat=num_replicates)

    def loglik(self, jsfs):
        return self.momi.loglik(params=self.params, jsfs=jsfs)

    def grad(self, jsfs):
        v, g = self.momi.loglik_with_gradient(params=self.params, jsfs=jsfs)
        return float(v), {i: float(g[i]) for i in g}

    def time_loglik(self, jsfs, num_replicates):
        return self.time(
            lambda: self.momi.loglik(params=self.params, jsfs=jsfs), num_replicates
        )

    def time_grad(self, jsfs, num_replicates):
        return self.time(
            lambda: self.momi.loglik_with_gradient(params=self.params, jsfs=jsfs),
            num_replicates,
        )


def randomly_drop_sfs_entries(jsfs: COO, reduce: float, seed: int = None):
    # randomly (by some weight) makes sfs more sparse by dropping some entries
    # sfs entries are inversely propotional to their value
    np.random.seed(seed)
    coords = np.array(jsfs.coords)
    vals = jsfs.data
    coords_weight = coords.sum(0)
    coords_weight = np.exp(coords_weight.max() - coords_weight)
    coords_weight /= coords_weight.sum()

    nnz = coords.shape[1]
    ind = np.random.choice(
        range(nnz), size=int(nnz * reduce), replace=False, p=coords_weight
    )

    new_coords = tuple(coords[:, ind])
    new_vals = vals[ind]

    return sparse.COO(new_coords, new_vals, shape=jsfs.shape)


class Moments_Gradient_Comparison:
    def __init__(self, demo, sampled_demes, sample_sizes, jitted=True):
        self.momi = Momi(demo, sampled_demes, sample_sizes, jitted=jitted)
        self.params = Params(self.momi)
        self.sampled_demes = sampled_demes
        self.sample_sizes = sample_sizes

        sampled_demes = self.sampled_demes
        sample_sizes = self.sample_sizes
        demo_dict = self.params.demo_dict

        def loglik_moments_with_params(
            theta_train,
            theta_nuisance,
            G_ts,
            jsfs,
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
            if isinstance(jsfs, jnp.ndarray | np.ndarray):
                pass
            else:
                jsfs = jsfs.todense()
            jsfs = jsfs.flatten()[1:-1]
            return multinomial_log_likelihood(jsfs, esfs, esfs.sum())

        self.loglik_moments_with_params = loglik_moments_with_params

    def set_train(self, keys: list):
        [self.params.set_train(key, True) for key in keys]

    def grad_moments(self, jsfs, epsilon=1.0):
        theta_train = self.params._theta_train
        theta_nuisance = self.params._theta_nuisance
        G_ts = self.params._loglik_G_ts
        loglik_moments_with_params = self.loglik_moments_with_params
        return approx_fprime(
            theta_train, loglik_moments_with_params, epsilon, theta_nuisance, G_ts, jsfs
        )

    def grad_momi(self, jsfs):
        return self.momi.loglik_with_gradient(self.params, jsfs=jsfs)[1]

    def find_the_closest_numerical_gradient(self, jsfs):
        a_grad = self.grad_momi(jsfs)

        def f(epsilon):
            n_grad = self.grad_moments(jsfs, epsilon=epsilon)
            return np.square(a_grad - n_grad).sum()

        eps_hat = scipy.optimize.minimize_scalar(
            f, bounds=(1e-8, 0.5), method="bounded"
        ).x

        return self.grad_moments(jsfs, epsilon=eps_hat)

    def compare_gradients(self, jsfs):
        a_grad = self.grad_momi(jsfs)
        n_grad = self.find_the_closest_numerical_gradient(jsfs)
        assertion_message = f"momi3_grad = {[float(i) for i in a_grad]}, numerical_grad_moments = {n_grad} "
        assert np.allclose(
            a_grad, n_grad, atol=0, rtol=GRADIENT_RTOL
        ), assertion_message


# Gradient Tests
# Compare analytical grad of momi3 with numerical gradient of moments


def test_gradient_single_exponential_growth():
    (
        demo,
        _,
    ) = SingleDeme.Exponential().base()
    sampled_demes = ["A"]
    sample_sizes = [5]
    mgc = Moments_Gradient_Comparison(demo, sampled_demes, sample_sizes)
    mgc.set_train(["eta_0", "eta_1", "eta_2", "tau_1"])
    jsfs = jnp.array([0, 5, 4, 3, 2, 0])
    mgc.compare_gradients(jsfs)


def test_gradient_two_pop_pulse(run_type="pytest", **kwargs):
    (
        demo,
        _,
    ) = TwoDemes.Constant().pulse()
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 3]
    mgc = Moments_Gradient_Comparison(demo, sampled_demes, sample_sizes)
    mgc.set_train(["eta_0", "pi_0", "tau_1"])
    jsfs = jnp.array(
        [[0, 5, 4, 3], [2, 1, 3, 3], [1, 0, 1, 2], [3, 3, 0, 0], [0, 2, 4, 0]]
    )
    mgc.compare_gradients(jsfs)


def test_gradient_two_pop_migration(run_type="pytest", **kwargs):
    (
        demo,
        _,
    ) = TwoDemes.Constant().migration()
    sampled_demes = ["A", "B"]
    sample_sizes = [4, 3]
    mgc = Moments_Gradient_Comparison(demo, sampled_demes, sample_sizes)
    mgc.set_train(["eta_0", "rho_0", "tau_1"])
    jsfs = jnp.array(
        [[0, 5, 4, 3], [2, 1, 3, 3], [1, 0, 1, 2], [3, 3, 0, 0], [0, 2, 4, 0]]
    )
    mgc.compare_gradients(jsfs)
