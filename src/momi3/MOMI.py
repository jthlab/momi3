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
from momi3.event_tree import ETBuilder
from momi3.JAX_functions import JAX_functions
from momi3.lineage_sampler import bound_sampler
from momi3.optimizers import ProjectedGradient_optimizer
from momi3.Params import Params
from momi3.utils import msprime_chromosome_simulator, msprime_simulator, tqdm, bootstrap_sample


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
        low_memory: bool = False,
    ):
        """
        Args:
            demo (demes.Graph): Demes graph
            sampled_demes (tuple[str]): Order of the demes.
                The order should match sample_sizes and jsfs dimensions.
            sample_sizes (tuple[int]): Sample sizes.
            jitted (bool, optional): Whether jax.jit the functions or not
            batch_size (int, optional): memory chunk size for each batch sfs entries
            low_memory (bool, optional): Will wrap the batch loop in jax.checkpoint
        """
        self._demo = demo.in_generations()
        self._jitted = jitted
        self._lowmem = low_memory
        assert len(sampled_demes) == len(sample_sizes)
        self.sampled_demes = tuple(sampled_demes)
        self.sample_sizes = tuple(sample_sizes)
        self._n_samples = dict(zip(sampled_demes, sample_sizes))
        self.batch_size = batch_size
        self._T = ETBuilder(self._demo, self._n_samples)
        self._JAX_functions = JAX_functions(
            demo=self._demo, T=self._T, jitted=self._jitted, low_memory=self._lowmem
        )

    @property
    def demo(self):
        return self._demo

    @property
    def _default_params(self):
        return Params(demo_dict=self._demo.asdict(), T=self._T)

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

    def sfs_spectrum(self, params="default"):
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
        esfs_vec = esfs(theta_dict, num_deriveds, self.batch_size)

        spectrum = np.zeros([self._n_samples[pop] + 1 for pop in self._n_samples])
        for b, val in zip(mutant_sizes[1:-1], esfs_vec[1:-1]):
            spectrum[tuple(b)] = val
        return spectrum

    def loglik(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        theta_train_dict: dict[str, float] = None,
    ) -> float:
        """log likelihood value for given params and data. It calls loglik_with_gradient.

        Args:
            params: Parameter values. Call Params(momi) to create one
            jsfs: Joint Site Frequency Spectrum. A n-dimensional array, where n is the number of leaf demes
            theta_train_dict: Dictionary for training values. See params._theta_train_dict

        Returns:
            float: log-likelihood value
        """

        ttd = params.theta_train_dict()
        tpd = params._theta_path_dict

        if theta_train_dict is None:
            theta_train_dict = ttd
        else:
            assert set(theta_train_dict) == set(ttd)

        theta_train_path_dict = {
            self._key_to_paths(
                key, params, False
            ): theta_train_dict[key] for key in theta_train_dict}

        data = self._get_data(jsfs)
        v = self._JAX_functions.loglik(theta_train_path_dict, tpd, data)
        return float(v)

    def loglik_with_gradient(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        theta_train_dict: dict[str, float] = None,
        transformed: bool = False,
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

        ttd = params.theta_train_dict(transformed)
        tpd = params._theta_path_dict

        if theta_train_dict is None:
            theta_train_dict = ttd
        else:
            assert set(theta_train_dict) == set(ttd)

        if transformed:
            theta_train_path_dict = ({}, {}, {}, {})
            for tkey in theta_train_dict:
                paths = self._key_to_paths(tkey, params, transformed)
                pkeys = ['eta', 'rho', 'pi', 'tau']
                for i, pkey in enumerate(pkeys):
                    if tkey.find(pkey) != -1:
                        theta_train_path_dict[i][paths] = theta_train_dict[tkey]
        else:
            theta_train_path_dict = {
                self._key_to_paths(
                    key, params, transformed
                ): theta_train_dict[key] for key in theta_train_dict}

        data = self._get_data(jsfs)
        V, G = self._JAX_functions.loglik_and_grad(
            theta_train_path_dict, tpd, data, transformed=transformed
        )

        if transformed:
            G = G[0] | G[1] | G[2] | G[3]

        G = {
            self._paths_to_keys(paths, params, transformed): float(G[paths]) for paths in G
        }

        return float(V), G

    def negative_loglik_with_gradient(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        theta_train_dict: dict[str, float] = None,
        transformed: bool = False,
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
            params, jsfs, theta_train_dict, transformed
        )

        val = -val
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
    ):

        tpd = params._theta_path_dict
        ttpd = params._theta_train_path_dict()

        data = self._get_data(jsfs)

        H_dict = self._JAX_functions.hessian(
            ttpd, tpd, data
        )
        H = []
        for i in ttpd:
            row = []
            for j in ttpd:
                row.append(H_dict[i][j])
            H.append(row)
        H = jnp.array(H)

        if just_hess:
            return H

        G = self._JAX_functions.loglik_and_grad(
            ttpd, tpd, data
        )[1]
        G = jnp.array([G[i] for i in ttpd])
        J = jnp.outer(G, G)
        J_inv = jnp.linalg.pinv(J)  # Calling psuedo-inverse

        return jnp.dot(jnp.dot(H, J_inv), H)

    def GIM_uncert(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        return_COV_MATRIX: bool = False,
    ):
        GIM = self.GIM(params, jsfs)
        COV = jnp.linalg.pinv(GIM)  # Calling psuedo-inverse
        if return_COV_MATRIX:
            return COV
        else:
            std = jnp.sqrt(jnp.diag(COV))
            return {key: float(std[i]) for i, key in enumerate(params._train_keys)}

    def FIM_uncert(
        self,
        params: Params,
        jsfs: Union[COO, jnp.ndarray, np.ndarray],
        return_COV_MATRIX: bool = False
    ):
        H = self.GIM(params, jsfs, just_hess=True)
        COV = jnp.linalg.pinv(H)  # Calling psuedo-inverse
        if return_COV_MATRIX:
            return COV
        else:
            std = jnp.sqrt(1 / jnp.abs(jnp.diag(H)))
            return {key: std[i] for i, key in enumerate(params._train_keys)}

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
            seed=seed
        )

    def simulate_chromosome(self, sequence_length, recombination_rate, mutation_rate, n_samples=None, params=None, seed=None, low_memory=True):
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
        return msprime_chromosome_simulator(
            demo=demo,
            sampled_demes=sampled_demes,
            sample_sizes=sample_sizes,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            mutation_rate=mutation_rate,
            seed=seed,
            low_memory=low_memory
        )

    def simulate_human_genome(self, recombination_rate=1e-8, mutation_rate=1e-8, seed=None):
        chr_lengths = [
            248956422,
            242193529,
            198295559,
            190214555,
            181538259,
            170805979,
            159345973,
            145138636,
            138394717,
            133797422,
            135086622,
            133275309,
            114364328,
            107043718,
            101991189,
            90338345,
            83257441,
            80373285,
            58617616,
            64444167,
            46709983,
            50818468
        ]

        jsfs = np.zeros([i + 1 for i in self.sample_sizes])
        for chr_length in tqdm(chr_lengths):
            seed = np.random.default_rng(seed).integers(2**32)
            jsfs += self.simulate_chromosome(
                chr_length, 1e-8, 1e-8, seed=seed, low_memory=False
            )
        return jsfs

    def bound_sampler(
        self,
        params: Params,
        size: int,
        scale: dict[str, float] = None,
        seed: int = None,
        quantile: float = 0.95,
        min_lineages: int = 4
    ):
        loc = params._theta_train
        if scale is None:
            scale = len(loc) * [0]
        else:
            scale = [scale[i] for i in params._train_keys]
        return bound_sampler(
            T=self._T,
            params=params,
            size=size,
            loc=loc,
            scale=scale,
            seed=seed,
            quantile=quantile,
            min_lineages=min_lineages
        )

    def _bootstrap_sample(
        self, jsfs: Union[COO, jnp.ndarray, np.ndarray], n_SNPs: int = None, seed=None
    ):

        return bootstrap_sample(jsfs, n_SNPs, seed)

    def _key_to_paths(self, key, params, transformed):
        if transformed:
            key = params._transforms_to_params[key]
            if isinstance(key, tuple):
                ret = (params._params_to_paths[key[0]], params._params_to_paths[key[1]])
            else:
                ret = params._params_to_paths[key]
        else:
            ret = params._params_to_paths[key]

        return ret

    def _paths_to_keys(self, paths, params, transformed):
        if transformed:
            try:
                key = params._paths_to_params[paths]
            except:
                key = params._paths_to_params[paths[0]], params._paths_to_params[paths[1]]

            return params._params_to_transforms[key]
        else:
            return params._paths_to_params[paths]

    def bound(self, bounds):
        self._T = ETBuilder(self._demo, self._n_samples).bound(bounds)
        self._JAX_functions = JAX_functions(
            demo=self._demo, T=self._T, jitted=self._jitted, low_memory=self._lowmem
        )
        return self

    def _get_data(self, jsfs):
        return get_data(
            self.sampled_demes, self.sample_sizes, self._T._leaves, jsfs, self.batch_size
        )

    def _time_loglik(self, params, jsfs, repeat=25, average=True):
        vals = {"val": 0}

        def f():
            return vals.update({"val": self.loglik(params=params, jsfs=jsfs)})

        compilation_time = timeit.timeit(f, number=1)
        run_time = timeit.repeat(f, repeat=repeat, number=1)
        if average:
            run_time = np.median(run_time)
        return vals["val"], compilation_time, run_time

    def _time_loglik_with_gradient(self, params, jsfs, repeat=25, average=True):
        vals = {"val": 0}

        def f():
            return vals.update(
                {"val": self.loglik_with_gradient(params=params, jsfs=jsfs)}
            )

        compilation_time = timeit.timeit(f, number=1)
        run_time = timeit.repeat(f, repeat=repeat, number=1)
        if average:
            run_time = np.median(run_time)
        return vals["val"], compilation_time, run_time
