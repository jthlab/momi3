from functools import partial
import numbers
from copy import deepcopy
from typing import Any, Callable
from collections.abc import Collection

from loguru import logger

import equinox as eqx
import demes
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import xlogy


from momi3.common import Path, set_path
from momi3.iicr.event_tree import IicrEventTree
from momi3.jsfs import JSFS

# from momi3.lineage_sampler import bound_sampler
from momi3.sfs.event_tree import SfsEventTree


def _set_path(d, path, value):
    for i in path[:-1]:
        d = d[i]
    d[path[-1]] = value


def _update_from_paths(d, paths):
    d = deepcopy(d)
    for path in paths:
        _set_path(d, path, paths[path])
    return d


class Momi3:
    def __init__(self, demo: demes.Graph):
        """
        Initialize the MOMI3 object.

        Args:
            demo: demes.Graph object representing the demography.
            n_samples: Dictionary mapping deme names to sample sizes.
        """
        if demo.time_units != "generations":
            raise ValueError(
                "Time units must be in generations. "
                "Please use demes.Graph.in_generations() to convert."
            )
        self._demo = demo

    @property
    def demo(self):
        return self._demo

    def sfs(self, num_samples: dict[str, int]):
        return _Momi3Sfs(self._demo, num_samples)

    def iicr(self, n: dict[str, int]):
        return _Momi3Iicr(self._demo, n)

    def coalescence_rate_trajectory(
        self,
        t: jax.Array,
        lineages: dict[str, int],
        _nd: bool = False,
        _jit: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Convencience function to compute the coalescence rate trajectory for a given set of lineages.

        Args:
            t: Array of times at which to compute the coalescence rate.
            lineages: Dictionary mapping deme names to the number of sampled lineages.

        Returns:
            Tuple of arrays: (coalescence rate, survival function)

        Note:
            This function mirrors the `coalescence_rate_trajectory` method of the `msprime.DemographyDebugger` class.
        """
        n = sum(lineages.values())
        f = partial(self.iicr(n), lineages)
        f = vmap(f)
        if _jit:
            f = jit(f)
        c, log_s = f(t)
        return c, jnp.exp(log_s)


class _Momi3Base:
    def __init__(self, demo: demes.Graph):
        self._demo = demo
        self._params_d = jax.tree.map(
            lambda v: float(v) if isinstance(v, numbers.Number) else v, demo.asdict()
        )

    @property
    def aux(self):
        return self._aux

    @property
    def params(self):
        return self._demo.asdict()

    def reparameterize(self, paths: Collection[Path]) -> tuple[Callable, Callable]:
        r"""
        Bijectively map demographic parameters to R^d.

        Args:
            params: list of paths to reparameterize.

        Notes:
            The reparameterization obeys necessary constraints on the demography. For example,
            consider the following isolation-with-pulse migration model:

                                      anc
                                       |
                                      .-.   <--- t_div
                                    /     \
                                   /       \
                                  /         \
                                 / <---p---- \ <--- t_pulse
                                /             \
                               A               B

            Here t_pulse must be less than t_div, p \in [0, 1], and t_div > 0. The reparameterization
            will ensure these constraints are met for any x \in R^3.

        Returns:
            Tuple of functions: (forward, inverse)
            The forward function maps from the original parameter space to R^d.
            The inverse function maps from R^d back to the original parameter space.

        Notes:
            The forward function contains an attribute `constraints` which explains the constraints
            on the reparameterized space.

        Example:
            >>> momi = Momi3(demo)  # demo corresponds to the demography above
            >>> forward, inverse = momi.reparameterize([('pulses', 0, 'time'), ('pulses', 0, 'proportion')])
            >>> forward.constraints
            {'pulses': {'time': 't_div', 'proportion': 'p'}}

        """
        return self._T.reparameterize(paths)


class _Momi3Iicr(_Momi3Base):
    def __init__(self, demo: demes.Graph, n: int):
        super().__init__(demo)
        self._n = n
        self._T = IicrEventTree(self._demo, n)
        self._aux = self._T.setup()

    def __call__(
        self, num_samples: dict[str, int], t: float, params: dict[Path, int] = {}
    ) -> float:
        if not num_samples.keys() <= set(self._T.leaves):
            setdiff = set(num_samples) - set(self._T.leaves)
            raise ValueError(
                f"Some sampled populations do not exist in the demography: {setdiff}. "
                f"Demography populations are: {list(self._T.leaves)}"
            )
        pred = sum(num_samples.values()) != self._n
        num_samples = eqx.error_if(
            num_samples, pred, f"Number of lineages must equal n={self._n}."
        )
        pd = _update_from_paths(self._params_d, params)
        for path in params:
            _set_path(pd, path, params[path])
        return self._T.execute(params=pd, num_samples=num_samples, t=t, aux=self._aux)

    def ET(self, params: dict[Path, int] = {}) -> float:
        "Expected time to first coalescence"

        def f(t):
            return self.sf(t, params)

        t_max = 1.0
        while f(t_max) > 1e-7:
            t_max *= 2

        t = jnp.linspace(0, t_max, 1000)
        return jnp.trapezoid(vmap(f)(t), t)


class _Momi3Sfs(_Momi3Base):
    def __init__(
        self, demo: demes.Graph, num_samples: dict[str, int], _event_tree=None
    ):
        super().__init__(demo)
        self._bounded = False
        self._num_samples = num_samples
        if _event_tree is not None:
            self._event_tree = _event_tree
        else:
            self._event_tree = SfsEventTree(self._demo, self._num_samples)
        if not (set(num_samples) <= set(self._event_tree.leaves)):
            setdiff = set(num_samples) - set(self._event_tree.leaves)
            raise ValueError(
                f"Some sampled populations do not exist in the demography: {setdiff}"
            )
        # self._params = Params(demo=self._demo, T=self._T)
        with jax.disable_jit(True):
            self._aux = self._event_tree.setup()

    @property
    def sampled_demes(self):
        return list(self._num_samples.keys())

    def E_tbl(
        self,
        path_d: dict[Path, float],
        num_derived: dict[str, int],
        aux,
    ) -> float:
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
        for pop in self._event_tree.leaves:
            # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
            n = num_samples.get(pop, 0)
            d = num_derived.get(pop, 0)
            # checkify.check(d <= n, f"More derived alleles than samples in {pop}")
            X[pop] = jax.nn.one_hot(jnp.array([d]), n + 1)[0]
        pd = deepcopy(self.params)
        for path, val in path_d.items():
            set_path(pd, path, val)
        return self._event_tree.execute(pd, X, aux).phi

    def E_tau(self, path_d: dict[Path, float], aux: Any) -> float:
        """Compute the expected total branch length of the genealogy for a given set of parameters.

        Args:
            path_d: A dictionary of parameter values.

        Returns:
            Expected total branch length subtending the given configuration.
        """
        X_batch = {}
        for pop in self._event_tree.leaves:
            ns = self._num_samples.get(pop, 0)
            X_batch[pop] = jnp.array(
                [
                    jnp.ones(ns + 1, dtype="f"),
                    jax.nn.one_hot(jnp.array([0]), ns + 1)[0],
                    jax.nn.one_hot(jnp.array([ns]), ns + 1)[0],
                ]
            )
        pd = deepcopy(self.params)
        for path, val in path_d.items():
            set_path(pd, path, val)
        phi = vmap(self._event_tree.execute, in_axes=(None, 0, None))(
            pd, X_batch, aux
        ).phi
        return phi[0] - phi[1] - phi[2]

    def expected_sfs(
        self,
        path_d: dict[Path, float] = {},
        aux: Any = None,
        _use_vmap: bool = True,
        _batch_size: int = None,
    ):
        if _batch_size is not None and _use_vmap:
            logger.warning("Batch size is ignored when using vmap")
        if aux is None:
            aux = self._aux
        bs = [n + 1 for n in self._num_samples.values()]
        num_derived = jnp.indices(bs)
        num_derived = jnp.rollaxis(num_derived, 0, num_derived.ndim).reshape(
            -1, len(bs)
        )

        @jit
        def f(ds):
            d = dict(zip(self._num_samples, ds))
            return self.E_tbl(path_d, d, self._aux)

        if _use_vmap:
            etbls = vmap(f)(num_derived)
        else:
            etbls = lax.map(f, num_derived, batch_size=_batch_size)
        tau = self.E_tau(path_d, aux=aux)
        sh = tuple(n + 1 for n in self._num_samples.values())
        return etbls.reshape(sh), tau

    def loglik(
        self,
        path_d: dict[Path, float],
        jsfs: JSFS,
        *,
        theta: float = None,
        folded: bool = True,
        use_vmap: bool = True,
        aux=None,
    ) -> float:
        """Log likelihood of joint site frequency spectrum.

        Args:
            path_d: Mapping of parameter keys to values.
            jsfs: Joint Site Frequency Spectrum, represented as a sparse tensor.
                The size of each axis should be 1 + sample size, with axis ordering
                corresponding to the key ordering in self.num_samples.
            theta: mutation rate per unit time, if known.
            folded: Conduct inference on the folded allele frequency spectrum.

        Returns:
            float: log-likelihood value
        """
        if aux is None:
            aux = self._aux
        f = self._loglik_vmap if use_vmap else self._loglik_scan
        return f(path_d, jsfs, folded, aux=aux, theta=theta)[0]

    def _configs(self, ds: list[int], folded: bool):
        ns = jnp.array(list(self._num_samples.values()))
        if folded:
            ds = ns - ds
        return dict(zip(self._num_samples, ds))

    def _branch_lengths(
        self, path_d: dict[Path, float], jsfs: JSFS, folded: bool, aux
    ) -> float:
        configs = [vmap(lambda ds: self._configs(ds, False))(jsfs.sites)]

        if folded:
            configs.append(vmap(lambda ds: self._configs(ds, True))(jsfs.sites))

        configs = jax.tree.map(lambda *a: jnp.stack(a, axis=1).reshape(-1), *configs)

        # encode as one-hot
        X = jax.tree.map(
            lambda ds, ns: jax.nn.one_hot(ds, ns + 1), configs, self._num_samples
        )

        # total branch length calcs
        X_tau = {}
        for pop, ns in self._num_samples.items():
            ns = self._num_samples.get(pop, 0)
            X_tau[pop] = jnp.array(
                [
                    jnp.ones(ns + 1, dtype="f"),
                    jax.nn.one_hot(jnp.array([0]), ns + 1)[0],
                    jax.nn.one_hot(jnp.array([ns]), ns + 1)[0],
                ],
                dtype=int,
            )

        # merge together all configs
        X_batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), X, X_tau)
        pd = deepcopy(self.params)
        for path, val in path_d.items():
            set_path(pd, path, val)
        etbls = vmap(self._event_tree.execute, in_axes=(None, 0, None))(
            pd, X_batch, aux
        ).phi
        tau = jax.tree.map(lambda e: e[-3] - e[-2] - e[-1], etbls).clip(2e-10)
        etbls = jax.tree.map(lambda e: e[:-3], etbls).clip(1e-10)

        if folded:
            etbls = jax.tree_map(lambda e: e.reshape(-1, 2).sum(axis=1), etbls)

        return etbls, tau

    def _loglik_vmap(
        self,
        path_d: dict[Path, float],
        jsfs: JSFS,
        folded: bool,
        aux,
        theta: float = None,
    ) -> float:
        etbls, tau = self._branch_lengths(path_d, jsfs, folded, aux)
        p = jsfs.counts  # / jsfs.counts.sum()
        # jax.debug.print('etbls: {}', etbls)
        # jax.debug.print('tau: {}', tau)
        if theta is not None:
            e = etbls * theta
            ll = jnp.sum(-e + xlogy(jsfs.counts, e))
        else:
            ll = xlogy(p, etbls / tau).sum()
        return ll, tau

    def _loglik_scan(
        self,
        path_d: dict[Path, float],
        jsfs: JSFS,
        folded: bool,
        aux,
        theta: float = None,
    ) -> float:
        tau = self.E_tau(path_d, aux=aux)

        def f(accum, tup):
            ds, pi = tup
            confs = [self._configs(ds, False)]
            if folded:
                confs.append(self._configs(ds, True))
            etbl = jnp.array([self.E_tbl(path_d, c, aux=aux) for c in confs]).mean()
            accum += xlogy(pi, etbl / tau)
            return accum, None

        p = jsfs.counts  # / jsfs.counts.sum()
        ll = lax.scan(f, 0.0, (jsfs.sites, p))[0]
        return ll, tau

    # def bound(
    #     self,
    #     size: int,
    #     scale: dict[str, float] = {},
    #     seed: int = None,
    #     quantile: float = 0.95,
    #     min_lineages: int = 4,
    # ):
    #     if hasattr(self, "_bounded"):
    #         raise ValueError("Already bounded")
    #     train_keys = self.params.trainable
    #     pd = {key: self.params[key].value for key in train_keys}
    #     if not self.constraints.valid(pd):
    #         raise ValueError("Initial parameters violate constraints")
    #     loc = list(pd.values())
    #     scale = [scale.get(k, 0.0) for k in train_keys]
    #     bounds = bound_sampler(
    #         T=self._T,
    #         # the sampler makes many changes to params, so have it work on a copy
    #         params=deepcopy(self.params),
    #         size=size,
    #         loc=loc,
    #         scale=scale,
    #         seed=seed,
    #         quantile=quantile,
    #         min_lineages=min_lineages,
    #     )
    #     logger.debug("bounds:{}", bounds)
    #     self._T = ETBuilder(self._demo, self._num_samples).bound(bounds)
    #     self._bounded = True
