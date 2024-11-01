import itertools as it
from copy import deepcopy

import demes
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import xlogy

from momi3.common import Path
from momi3.jsfs import JSFS

# from momi3.lineage_sampler import bound_sampler
from momi3.sfs.event_tree import SfsEventTree


def _set_path(d, path, value):
    for i in path[:-1]:
        d = d[i]
    d[path[-1]] = value


class Momi3:
    def __init__(self, demo: demes.Graph, num_samples: dict[str, int]):
        """
        Initialize the MOMI3 object.

        Args:
            demo: demes.Graph object representing the demography.
            n_samples: Dictionary mapping deme names to sample sizes.
        """
        self._demo = demo
        self._params_d = demo.asdict()
        self._num_samples = num_samples
        self._T = SfsEventTree(self._demo, self._num_samples)
        self._aux = self._T._setup()
        if not (set(num_samples) <= set(self._T.leaves)):
            setdiff = set(num_samples) - set(self._T.leaves)
            raise ValueError(
                f"Some sampled populations do not exist in the demography: {setdiff}"
            )
        # self._params = Params(demo=self._demo, T=self._T)

    @property
    def demo(self):
        return self._demo

    # @property
    # def params(self) -> dict[str, float]:
    #     """Default parameters specified by demography"""
    #     return self._params

    # @property
    # def constraints(self):
    #     return self.params.constraints

    def E_tbl(
        self,
        params: dict[Path, float],
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
        for pop in self._T.leaves:
            # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
            n = num_samples.get(pop, 0)
            d = num_derived.get(pop, 0)
            # checkify.check(d <= n, f"More derived alleles than samples in {pop}")
            X[pop] = jax.nn.one_hot(jnp.array([d]), n + 1)[0]
        pd = deepcopy(self._params_d)
        for path in params:
            _set_path(pd, path, params[path])
        return self._T.execute(pd, X, aux)

    def E_tau(self, params: dict[Path, float], aux=None) -> float:
        """Compute the expected total branch length of the genealogy for a given set of parameters.

        Args:
            params_d: A dictionary of parameter values.

        Returns:
            Expected total branch length subtending the given configuration.
        """
        aux = aux or self._aux
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
        pd = deepcopy(self._params_d)
        for path in params:
            _set_path(pd, path, params[path])
        ret = vmap(self._T.execute, in_axes=(None, 0, None))(pd, X_batch, aux)
        return ret[0] - ret[1] - ret[2]

    def expected_sfs(self, params_d: dict[str, float] = {}, _use_vmap: bool = True):
        bs = [range(n + 1) for n in self._num_samples.values()]
        num_derived = jnp.array(list(it.product(*bs)))

        def f(ds):
            d = dict(zip(self._num_samples, ds))
            return self.E_tbl(params_d, dict(d), self._aux)

        if _use_vmap:
            etbls = vmap(f)(num_derived)
        else:
            etbls = lax.map(f, num_derived)
        tau = self.E_tau(params_d)
        sh = tuple(n + 1 for n in self._num_samples.values())
        return (etbls / tau).reshape(sh)

    def sf(self, t: float, params_d: dict[str, float] = {}):
        if sum(self._num_samples.values()) != 2:
            raise ValueError(
                "I only know how to compute the IICR for samples of size n=2"
            )
        r = jax.jacfwd(self.E_tbl, argnums=(3,))(
            params_d, self._num_samples, self._T.auxd, t
        )[0].squeeze()
        return 1.0 - r

    def iicr(self, t: float, params_d: dict[str, float] = {}):
        def f(t):
            return -jnp.log(self.sf(t, params_d))

        return jax.jacfwd(f)(t).squeeze()

    def loglik(
        self,
        params_d: dict[str, float],
        jsfs: JSFS,
        *,
        theta: float = None,
        folded: bool = True,
        use_vmap: bool = True,
        aux=None,
    ) -> float:
        """Log likelihood of joint site frequency spectrum.

        Args:
            params_d: Mapping of parameter keys to values.
            jsfs: Joint Site Frequency Spectrum, represented as a sparse tensor.
                The size of each axis should be 1 + sample size, with axis ordering
                corresponding to the key ordering in self.num_samples.
            theta: mutation rate per unit time, if known.
            folded: Conduct inference on the folded allele frequency spectrum.

        Returns:
            float: log-likelihood value
        """
        f = self._loglik_vmap if use_vmap else self._loglik_scan
        ll, _ = f(params_d, jsfs, folded, aux=aux, theta=theta)
        return ll

    def _configs(self, ds: list[int], folded: bool):
        ns = jnp.array(list(self._num_samples.values()))
        if folded:
            ds = ns - ds
        return dict(zip(self._num_samples, ds))

    def _branch_lengths(
        self, params_d: dict[str, float], jsfs: JSFS, folded: bool, aux
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

        pd = self.params.update(params_d).to_path_dict()
        etbls = vmap(self._T.execute, in_axes=(None, 0, None))(pd, X_batch, aux)

        tau = jax.tree.map(lambda e: e[-3] - e[-2] - e[-1], etbls).clip(2e-10)
        etbls = jax.tree.map(lambda e: e[:-3], etbls).clip(1e-10)

        if folded:
            etbls = jax.tree_map(lambda e: e.reshape(-1, 2).sum(axis=1), etbls)

        return etbls, tau

    def _loglik_vmap(
        self,
        params_d: dict[str, float],
        jsfs: JSFS,
        folded: bool,
        aux,
        theta: float = None,
    ) -> float:
        etbls, tau = self._branch_lengths(params_d, jsfs, folded, aux)
        p = jsfs.counts  # / jsfs.counts.sum()
        # jax.debug.print('etbls: {}', etbls)
        # jax.debug.print('tau: {}', tau)
        if theta is not None:
            e = etbls * theta
            ll = jnp.sum(-e + jsfs.counts * jnp.log(e))
        else:
            ll = xlogy(p, etbls / tau).sum()
        return ll, tau

    def _loglik_scan(
        self, params_d: dict[str, float], jsfs: JSFS, tau: float, folded: bool, aux
    ) -> float:
        def f(accum, tup):
            ds, pi = tup
            confs = self._configs(ds, folded)
            etbl = jnp.array([self.E_tbl(params_d, c, aux=aux) for c in confs]).mean()
            accum += xlogy(pi, etbl / tau)
            return accum, None

        p = jsfs.counts  # / jsfs.counts.sum()
        return lax.scan(f, 0.0, (jsfs.sites, p))[0]

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
