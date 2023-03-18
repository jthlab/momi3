import timeit
from copy import deepcopy
from itertools import product
from typing import Union

# import logging
import demes
import jax.numpy as jnp
import numpy as np
from sparse._coo.core import COO

from momi3.Data import get_data
from momi3.event import ETBuilder, Node, Population
from momi3.JAX_functions import JAX_functions
from momi3.optimizers import ProjectedGradient_optimizer
from momi3.Params import Params
from momi3.surviving_lineage_samplers import bound_sampler
from momi3.utils import msprime_simulator


def esfs(g: demes.Graph, sample_sizes: dict[str, int]):
    """Compute the expected site frequency spectrum for a given demography and sample.

    Params:
        g: the demography, expressed as a `demes.Graph`.
        sample_sizes: mapping from deme names in `g` to their sample sizes. Demes which are present in
            `g` but missing in `sample_sizes` are assumed to be unsampled.

    Returns:
        Array with one axis per entry in `sample_sizes`, representing the expected site frequency spectrum
        under `g`.
    """
    return Momi(g, *zip(*sample_sizes.items())).sfs_spectrum()


class Momi(object):
    def __init__(
        self,
        demo: demes.Graph,
        sampled_demes: tuple[str],
        sample_sizes: tuple[int],
        jitted: bool = False,
        batch_size: int = None,
        bounds: dict[Node, dict[Population, int]] = None,
    ):
        """
        Args:
            demo (demes.Graph): Demes graph
            sampled_demes (tuple[str]): Order of the demes.
                The order should match sample_sizes and jsfs dimensions.
            sample_sizes (tuple[int]): Sample sizes.
            jitted (bool, optional): Whether jax.jit the functions or not
        """
        demo = demo.in_generations()
        assert len(sampled_demes) == len(sample_sizes)

        self.sampled_demes = tuple(sampled_demes)
        self.sample_sizes = tuple(sample_sizes)
        self._n_samples = dict(zip(sampled_demes, sample_sizes))

        T = ETBuilder(demo, self._n_samples, bounds)
        demo_dict = demo.asdict()

        # assert_message = f"keys of n_samples do not match the demo {set(n_samples)} =/= {set(T._leaves)}"
        # assert set(n_samples) == set(T._leaves), assert_message

        self._T = T
        self._auxd = T.auxd
        self._demo_dict = demo_dict
        self._JAX_functions = JAX_functions(demo=demo, T=T, jitted=jitted)

    @property
    def _default_params(self):
        return Params(demo_dict=self._demo_dict, T=self._T)

    def sfs_entry(self, num_derived: dict, params="default"):
        if params == "default":
            params = self._default_params

        theta_dict = deepcopy(params._theta_train_dict)
        theta_dict.update(params._theta_nuisance_dict)

        esfs = self._JAX_functions.esfs
        for pop in num_derived:
            num_derived[pop] = np.array([num_derived[pop]])
        return esfs(theta_dict, num_derived, 1)[0]

    def total_branch_length(self, params="default"):
        if params == "default":
            params = self._default_params

        theta_dict = deepcopy(params._theta_train_dict)
        theta_dict.update(params._theta_nuisance_dict)

        etbl = self._JAX_functions.etbl
        return etbl(theta_dict)

    def sfs_spectrum(self, params="default", batch_size: int = 10000):
        if params == "default":
            params = self._default_params
        theta_dict = deepcopy(params._theta_train_dict)
        theta_dict.update(params._theta_nuisance_dict)

        bs = [jnp.arange(self._n_samples[pop] + 1) for pop in self._n_samples]
        mutant_sizes = jnp.array(list(product(*bs)))

        num_deriveds = {}
        for i, pop in enumerate(self._n_samples):
            num_deriveds[pop] = mutant_sizes[:, i]

        esfs = self._JAX_functions.esfs
        esfs_vec = esfs(theta_dict, num_deriveds, batch_size)

        spectrum = np.zeros([self._n_samples[pop] + 1 for pop in self._n_samples])
        for b, val in zip(mutant_sizes[1:-1], esfs_vec[1:-1]):
            spectrum[tuple(b)] = val
        return spectrum

    def loglik(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        theta_train_dict: dict[str, float] = None,
        batch_size: int = 10000
    ) -> float:
        """log likelihood value for given params and data. It calls loglik_with_gradient.

        Args:
            params: Parameter values. Call Params(momi) to create one
            jsfs: Joint Site Frequency Spectrum. A n-dimensional array, where n is the number of leaf demes
            theta_train_dict: Dictionary for training values. See params._theta_train_dict

        Returns:
            float: log-likelihood value
        """

        if theta_train_dict is None:
            theta_train_dict = params._theta_train_dict
        else:
            # Change the keys from param keys to paths
            param_keys = sorted(list(theta_train_dict.keys()))
            assert set(param_keys) == set(params._train_keys)
            Paths = [params._params_to_paths[param_key] for param_key in param_keys]
            values = [float(theta_train_dict[param_key]) for param_key in param_keys]
            theta_train_dict = dict(zip(Paths, values))

        theta_nuisance_dict = params._theta_nuisance_dict
        data = self._get_data(jsfs, batch_size)
        return self._JAX_functions.loglik(
            theta_train_dict, theta_nuisance_dict, data
        )

    def loglik_with_gradient(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        theta_train_dict: dict[str, float] = None,
        return_array: bool = False,
        batch_size: int = 10000
    ) -> tuple[float, dict[str, float]]:
        """log likelihood value and the gradient for theta_train_dict for given params and jsfs

        Args:
            params: Parameter values. Call Params(momi) to create one
            jsfs: Joint Site Frequency Spectrum. A n-dimensional array, where n is the number of leaf demes
            theta_train_dict: Dictionary for training values. See params._theta_train_dict
            return_array: If true if will return tuple[float, jnp.ndarray] sorted by name of the variable

        Returns:
            tuple[float, dict[str, float]]: log-likelihood value and its gradient
        """

        if theta_train_dict is None:
            theta_train_dict = params._theta_train_dict
        else:
            # Change the keys from param keys to paths
            param_keys = sorted(list(theta_train_dict.keys()))
            assert set(param_keys) == set(params._train_keys)
            Paths = [params._params_to_paths[param_key] for param_key in param_keys]
            values = [float(theta_train_dict[param_key]) for param_key in param_keys]
            theta_train_dict = dict(zip(Paths, values))

        theta_nuisance_dict = params._theta_nuisance_dict
        data = self._get_data(jsfs, batch_size)
        val, grad = self._JAX_functions.loglik_and_grad(
            theta_train_dict, theta_nuisance_dict, data
        )
        grad = {params._paths_to_params[i]: grad[i] for i in grad}
        if return_array:
            grad = jnp.array([grad[i] for i in sorted(grad)])
        return val, grad

    def negative_loglik_with_gradient(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray] = None,
        theta_train_dict: dict[str, float] = None,
        return_array: bool = False,
        batch_size: int = 10000,
    ) -> tuple[float, dict[str, float]]:
        """Negative log likelihood value and the gradient for theta_train_dict for given params and jsfs.
        It calls loglik_with_gradient.

        Args:
            params: Parameter values. Call Params(momi) to create one
            jsfs: Joint Site Frequency Spectrum. A n-dimensional array, where n is the number of leaf demes
            theta_train_dict: Dictionary for training values. See params._theta_train_dict
            return_array: If true if will return tuple[float, jnp.ndarray] sorted by name of the variable

        Returns:
            tuple[float, dict[str, float]]: Negative log-likelihood value and its gradient
        """

        val, grad = self.loglik_with_gradient(
            params, jsfs, theta_train_dict, return_array, batch_size
        )

        val = -val

        if return_array:
            grad = -grad
        else:
            grad = {i: -grad[i] for i in grad}

        return val, grad

    def optimize(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        stepsize: float,
        maxiter: int,
        theta_train_dict_0: dict[str, float] = None,
        htol: float = 0,
        monitor_training: bool = False,
    ) -> dict:
        """Maximum Likelihood estimator for theta_train. Optimizer is a wrappper for jaxopt.ProjectedGradient.
        There are two ways to give data:
            (I): jsfs
            (II): freqs_matrix and num_deriveds

        Args:
            params: Parameter values
            jsfs: Joint Site Frequency Spectrum. A n-dimensional array, where n is the number of leaf demes
            stepsize: Step size for (projected) gradient descent algorithm.
            maxiter: Number of iterations.
            theta_train_dict_0: Initial values for optimization. If None, initial demes values will be used.
            htol: Tolerance in polyhedran projection. This will enforce the gradient steps be in the constrained space
                G @ theta_train_hat <= h - htol
            monitor_training: It will print the state of the gradient descent in each iteration

        Returns:
            dict: Optimization Results
                likelihood_ratio = -2 * (loglik_0 - loglik_n)
                loglik_0 = Log-likelihood value at theta_train_0
                loglik_n = Log-likelihood value at the end of the optimization
                loglik_grad = Gradients of log-likelihood at the end of the optimization
                params = Updated Params
                pg_state = jax.ProjectedGradient state
                theta_train_hat = Optimized values

        """

        negative_loglik_with_gradient = self.negative_loglik_with_gradient
        sampled_demes = self.sampled_demes

        return ProjectedGradient_optimizer(
            negative_loglik_with_gradient=negative_loglik_with_gradient,
            params=params,
            jsfs=jsfs,
            stepsize=stepsize,
            maxiter=maxiter,
            theta_train_dict_0=theta_train_dict_0,
            sampled_demes=sampled_demes,
            htol=htol,
            monitor_training=monitor_training,
        )

    def GIM(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray] = None,
        just_hess: bool = False,
        batch_size: int = 10000,
    ):
        theta_train_dict = params._theta_train_dict
        theta_nuisance_dict = params._theta_nuisance_dict
        data = self._get_data(jsfs, batch_size)

        H_dict = self._JAX_functions.hessian(
            theta_train_dict, theta_nuisance_dict, data
        )
        H = []
        for i in theta_train_dict:
            row = []
            for j in theta_train_dict:
                row.append(H_dict[i][j])
            H.append(row)
        H = jnp.array(H)

        if just_hess:
            return H

        G = self._JAX_functions.loglik_and_grad(
            theta_train_dict, theta_nuisance_dict, data
        )[1]
        G = jnp.array([G[i] for i in theta_train_dict])
        J = jnp.outer(G, G)
        J_inv = jnp.linalg.pinv(J)  # Calling psuedo-inverse

        return jnp.dot(jnp.dot(H, J_inv), H)

    def GIM_uncert(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        return_COV_MATRIX: bool = False,
        batch_size: int = 10000,
    ):
        GIM = self.GIM(params, jsfs, batch_size=batch_size)
        COV = jnp.linalg.pinv(GIM)  # Calling psuedo-inverse
        if return_COV_MATRIX:
            return COV
        else:
            return jnp.sqrt(jnp.diag(COV))

    def FIM_uncert(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        return_COV_MATRIX: bool = False,
        batch_size: int = 10000,
    ):
        FIM = self.GIM(params, jsfs, just_hess=True, batch_size=batch_size)
        COV = jnp.linalg.pinv(FIM)  # Calling psuedo-inverse
        if return_COV_MATRIX:
            return COV
        else:
            return jnp.sqrt(jnp.diag(COV))

    def simulate(self, num_replicates, n_samples=None, params=None, seed=None):
        if params is None:
            params = self._default_params

        if n_samples is None:
            sampled_demes = self.sampled_demes
            sample_sizes = self.sample_sizes
        else:
            assert set(n_samples) == set(self.sampled_demes)
            sampled_demes = list(n_samples.keys())
            sample_sizes = list(n_samples.values())

        demo = params.demo_graph

        return msprime_simulator(
            demo=demo,
            sampled_demes=sampled_demes,
            sample_sizes=sample_sizes,
            num_replicates=num_replicates,
            seed=seed,
        )

    def bound_sampler(
        self,
        params: Params,
        scale: np.ndarray,
        size: int,
        seed: int = None,
        quantile: float = 0.95,
    ):
        loc = params._theta_train
        return bound_sampler(
            T=self._T,
            params=params,
            size=size,
            loc=loc,
            scale=scale,
            seed=seed,
            quantile=quantile,
        )

    def _get_data(self, jsfs, batch_size):
        return get_data(
            self.sampled_demes, self.sample_sizes, self._T._leaves, jsfs, batch_size
        )

    def _time_loglik(self, params, jsfs, batch_size=10000, repeat=25, average=True):
        vals = {"val": 0}

        def f():
            return vals.update({"val": self.loglik(params=params, jsfs=jsfs, batch_size=batch_size)})

        compilation_time = timeit.timeit(f, number=1)
        run_time = timeit.repeat(f, repeat=repeat, number=1)
        if average:
            run_time = np.median(run_time)
        return vals["val"], compilation_time, run_time

    def _time_loglik_with_gradient(self, params, jsfs, batch_size=10000, repeat=25, average=True):
        vals = {"val": 0}

        def f():
            return vals.update(
                {"val": self.loglik_with_gradient(params=params, jsfs=jsfs, batch_size=batch_size)}
            )

        compilation_time = timeit.timeit(f, number=1)
        run_time = timeit.repeat(f, repeat=repeat, number=1)
        if average:
            run_time = np.median(run_time)
        return vals["val"], compilation_time, run_time
