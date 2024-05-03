import itertools as it
import timeit
from copy import deepcopy
from typing import OrderedDict, Union

import demes
import jax
import jax.numpy as jnp
import numpy as np
import sparse
from frozendict import frozendict
from jax import jit, lax, value_and_grad, vmap
from jax.scipy.special import xlogy
from loguru import logger

from momi3.Data import get_data
from momi3.event_tree import ETBuilder
from momi3.jsfs import JSFS
from momi3.lineage_sampler import bound_sampler
from momi3.optimizers import ProjectedGradient_optimizer
from momi3.Params import Params
from momi3.utils import (
    bootstrap_sample,
    msprime_chromosome_simulator,
    msprime_simulator,
    tqdm,
)


class Momi3:
    def __init__(self, demo: demes.Graph, num_samples: OrderedDict[str, int]):
        """
        Initialize the MOMI3 object.

        Args:
            demo: demes.Graph object representing the demography.
            n_samples: Dictionary mapping deme names to sample sizes.
        """
        self._demo = demo
        self._num_samples = num_samples
        self._T = ETBuilder(self._demo, self._num_samples)
        if not (set(num_samples) <= set(self._T.leaves)):
            setdiff = set(num_samples) - set(self._T.leaves)
            raise ValueError(
                f"Some sampled populations do not exist in the demography: {setdiff}"
            )
        self._params = Params(demo=self._demo, T=self._T)

    @property
    def demo(self):
        return self._demo

    @property
    def params(self) -> dict[str, float]:
        """Default parameters specified by demography"""
        return self._params

    @property
    def constraints(self):
        return self.params.constraints

    def E_tbl(self, params_d: dict[str, float], num_derived: dict[str, int]) -> float:
        """Compute the expected total branch length of the genealogy for a given set of parameters.

        Args:
            params_d: A dictionary of parameter values.
            num_derived: A dictionary mapping deme names to the number of derived alleles.

        Returns:
            Expected total branch length subtending the given configuration.
        """
        # require that the derived allele counts are consistent with the sample sizes
        num_samples = self._num_samples
        assert set(num_samples) == set(num_derived)
        # create X mapping each population to a one-hot encoded array of derived allele counts
        X = {}
        for pop in self._T.leaves:
            # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
            n = num_samples.get(pop, 0)
            d = num_derived.get(pop, 0)
            # checkify.check(d <= n, f"More derived alleles than samples in {pop}")
            X[pop] = jnp.eye(n + 1)[d]
        pd = self.params.update(params_d).to_path_dict()
        return self._T.execute(pd, X, self._T.auxd).clip(1e-10)

    def E_tau(self, params_d: dict[str, float]) -> float:
        """Compute the expected total branch length of the genealogy for a given set of parameters.

        Args:
            params_d: A dictionary of parameter values.

        Returns:
            Expected total branch length subtending the given configuration.
        """
        X_batch = {}
        for pop in self._T.leaves:
            ns = self._num_samples.get(pop, 0)
            X_batch[pop] = jnp.array(
                [
                    jnp.ones(ns + 1, dtype="f"),
                    jax.nn.one_hot(jnp.array([0]), ns + 1)[0],
                    jax.nn.one_hot(jnp.array([ns]), ns + 1)[0],
                ]
            )
        pd = self.params.update(params_d).to_path_dict()
        ret = vmap(self._T.execute, in_axes=(None, 0, None))(pd, X_batch, self._T.auxd)
        return ret[0] - ret[1] - ret[2]

    def expected_sfs(self, params_d: dict[str, float] = {}):
        bs = [range(n + 1) for n in self._num_samples.values()]
        num_derived = jnp.array(list(it.product(*bs)))

        @vmap
        def f(ds):
            d = dict(zip(self._num_samples, ds))
            return self.E_tbl(params_d, dict(d))

        etbls = f(num_derived)
        tau = self.E_tau(params_d)
        sh = tuple(n + 1 for n in self._num_samples.values())
        return (etbls / tau).reshape(sh)

    def loglik(
        self,
        params_d: dict[str, float],
        jsfs: JSFS,
        theta: float = None,
        use_vmap: bool = True,
    ) -> float:
        """Log likelihood of joint site frequency spectrum.

        Args:
            params_d: Mapping of parameter keys to values.
            jsfs: Joint Site Frequency Spectrum, represented as a sparse tensor.
                The size of each axis should be 1 + sample size, with axis ordering
                corresponding to the key ordering in self.num_samples.
            theta: mutation rate per unit time, if known.

        Returns:
            float: log-likelihood value
        """
        f = self._loglik_vmap if use_vmap else self._loglik_scan
        return f(params_d, jsfs, theta)

    def _loglik_vmap(
        self, params_d: dict[str, float], jsfs: JSFS, theta: float
    ) -> float:
        @vmap
        def f(ds):
            return self.E_tbl(params_d, dict(zip(self._num_samples, ds)))

        etbls = f(jsfs.sites)
        tau = self.E_tau(params_d)
        p = jsfs.counts / jsfs.counts.sum()
        ret = xlogy(p, etbls / tau).sum()
        return ret
        # if theta is not None:
        # # theta is not none, use the poisson likelihood
        #     ret += jax.scipy.stats.poisson.logpmf(jsfs.counts.sum(), theta * tau).sum()
        # ns = jsfs.nonseg_sites
        # return jnp.where(
        #     ns,
        #     -jsfs.counts * theta * tau,
        #     jax.scipy.stats.poisson.logpmf(jsfs.counts, theta * etbls),
        # ).sum()

    def _loglik_scan(self, params_d: dict[str, float], jsfs: JSFS) -> float:
        tau = self.E_tau(params_d)

        def f(accum, tup):
            ds, pi = tup
            etbl = self.E_tbl(params_d, dict(zip(self._num_samples, ds)))
            accum += xlogy(pi, etbl / tau)
            return accum, None

        p = jsfs.counts / jsfs.counts.sum()
        return lax.scan(f, 0.0, (jsfs.sites, p))[0]

    def optimize(
        self,
        params: Params,
        jsfs: JSFS,
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
            htol: Tolerance in polyhedron projection. This will enforce the gradient steps be in the constrained space
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
        jsfs: JSFS = None,
        just_hess: bool = False,
    ):
        tpd = params._theta_path_dict
        ttpd = params._theta_train_path_dict()

        data = self._get_data(jsfs)

        H_dict = self._JAX_functions.hessian(ttpd, tpd, data)
        H = []
        for i in ttpd:
            row = []
            for j in ttpd:
                row.append(H_dict[i][j])
            H.append(row)
        H = jnp.array(H)

        if just_hess:
            return H

        G = self._JAX_functions.loglik_and_grad(ttpd, tpd, data)[1]
        G = jnp.array([G[i] for i in ttpd])
        J = jnp.outer(G, G)
        J_inv = jnp.linalg.pinv(J)  # Calling psuedo-inverse

        return jnp.dot(jnp.dot(H, J_inv), H)

    def GIM_uncert(
        self,
        params: Params,
        jsfs: Union[sparse.COO, jnp.ndarray, np.ndarray],
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
        jsfs: JSFS,
        return_COV_MATRIX: bool = False,
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
            seed=seed,
        )

    def simulate_chromosome(
        self,
        sequence_length,
        recombination_rate,
        mutation_rate,
        n_samples=None,
        params=None,
        seed=None,
        low_memory=True,
    ):
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
            low_memory=low_memory,
        )

    def simulate_human_genome(
        self, recombination_rate=1e-8, mutation_rate=1e-8, seed=None
    ):
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
            50818468,
        ]

        jsfs = np.zeros([i + 1 for i in self.sample_sizes])
        for chr_length in tqdm(chr_lengths):
            seed = np.random.default_rng(seed).integers(2**32)
            jsfs += self.simulate_chromosome(
                chr_length, 1e-8, 1e-8, seed=seed, low_memory=False
            )
        return jsfs

    def bound(
        self,
        size: int,
        scale: dict[str, float] = {},
        seed: int = None,
        quantile: float = 0.95,
        min_lineages: int = 4,
    ):
        if hasattr(self, "_bounded"):
            raise ValueError("Already bounded")
        train_keys = self.params.trainable
        loc = [self.params[key].value for key in train_keys]
        scale = [scale.get(k, 0.0) for k in train_keys]
        bounds = bound_sampler(
            T=self._T,
            # the sampler makes many changes to params, so have it work on a copy
            params=deepcopy(self.params),
            size=size,
            loc=loc,
            scale=scale,
            seed=seed,
            quantile=quantile,
            min_lineages=min_lineages,
        )
        logger.debug("bounds:{}", bounds)
        self._T = ETBuilder(self._demo, self._num_samples).bound(bounds)
        self._bounded = True

    def _bootstrap_sample(self, jsfs: JSFS, n_SNPs: int = None, seed=None):
        return bootstrap_sample(jsfs, n_SNPs, seed)

    def _key_to_path(self, key, params, transformed):
        if transformed:
            key = params._transforms_to_params[key]
            if isinstance(key, tuple):
                return (
                    params._params_to_paths[key[0]],
                    params._params_to_paths[key[1]],
                )
            else:
                return params._params_to_paths[key]
        else:
            return params._params_to_paths[key]

    def _paths_to_keys(self, paths, params, transformed):
        if transformed:
            try:
                key = params._paths_to_params[paths]
            except IndexError:
                key = (
                    params._paths_to_params[paths[0]],
                    params._paths_to_params[paths[1]],
                )

            return params._params_to_transforms[key]
        else:
            return params._paths_to_params[paths]

    def _get_data(self, jsfs):
        return get_data(
            self.sampled_demes,
            self.sample_sizes,
            self._T._leaves,
            jsfs,
            self.batch_size,
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


if __name__ == "__main__":
    demo = demes.load("tests/yaml_files/IWM.yml")
    num_samples = frozendict({"deme0": 200, "deme1": 300})
    num_derived = {"deme0": 5, "deme1": 5}
    m3 = Momi3(demo, num_samples)
    bounds = m3.bound_sampler(100)
    print(bounds)
    p = m3.params
    print(p)
    pd = {"eta_0": 50.0}
    import logging

    logging.getLogger("jax").setLevel(logging.DEBUG)
    # etbl = jit(m3.E_tau)
    # detbl = jit(grad(etbl))
    # print(etbl(pd))
    # print(detbl(pd))
    loglik = jit(value_and_grad(m3.loglik))
    jsfs = JSFS(jnp.array([[5, 5]] * 100), jnp.array([10] * 100))
    print(loglik(pd, jsfs))

    import sys

    sys.exit(0)
