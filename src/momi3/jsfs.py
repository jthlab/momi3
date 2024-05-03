from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import sparse
from jaxtyping import Array, Int


class JSFS(NamedTuple):
    """Joint site frequency spectrum (JSFS), represent as a COO tensor.

    Attributes:
        sample_sizes: The sample sizes of each populations.
        coords: The coordinates of the nonzero entries.
        counts: The counts of the nonzero entries.

    Notes:
        The JSFS is a sparse tensor, where the dimensions are the sample sizes
        of the populations (plus one). The nonzero entries are the counts of the number of
        sites with a given frequency of derived alleles.
    """

    sample_sizes: dict[str, int]
    sites: Int[Array, "s d"]  # noqa: F722
    counts: Int[Array, "s"]

    def to_COO(self) -> sparse.COO:
        return sparse.COO(
            self.sites.T, self.counts, shape=tuple(s + 1 for s in self.sample_sizes)
        )

    @classmethod
    def from_COO(cls, coo: sparse.COO) -> "JSFS":
        return cls(
            sample_sizes=tuple(s - 1 for s in coo.shape),
            sites=coo.coords.T,
            counts=coo.data,
        )

    @property
    def s(self) -> int:
        return len(self.counts)

    @property
    def d(self) -> int:
        return len(self.sample_sizes)

    def random_sample(
        self,
        n: int,
        key: jax.random.PRNGKey,
        by_site: bool = True,
        uniform: bool = False,
    ) -> "JSFS":
        """Returns random sample of JSFS, where sites are drawn without replacement.

        Params:
            jsfs: joint-sfs
            n: number of SNPs/sites to sample
            seed: random seed, or 1 if None
            by_site: if True, sample by site, otherwise sample by SNP.
            uniform: if True, sample uniformly, otherwise sample by frequency.

        Returns:
            A sampled jsfs containing n_snps sites.
        """
        # sum of all entries
        if by_site:
            assert n <= self.s
            kw = dict(shape=(n,), replace=False)
            if not uniform:
                kw["p"] = self.counts / jnp.sum(self.counts)
            i = jax.random.choice(key, self.s, **kw)
            return self._replace(
                sites=jnp.array(self.sites)[i], counts=jnp.array(self.counts)[i]
            )
        else:
            assert not uniform
            c = jnp.cumsum(self.counts)
            i = jax.random.choice(key, c, shape=(n,), replace=False)
            j = jnp.searchsorted(c, i, side="right")
            return self._replace(
                sites=jnp.array(self.sites)[j], counts=jnp.ones_like(j)
            )

    @property
    def nonseg_sites(self) -> "JSFS":
        """Return a boolean mask indicating whether each site is segregating."""
        s1 = jnp.all(self.sites == jnp.zeros(self.d, dtype=int), axis=1)
        s2 = jnp.all(self.sites == jnp.array(self.sample_sizes.values()), axis=1)
        return s1 | s2

    def project(self, pops: list[str]) -> "JSFS":
        """Projects the jsfs onto the given populations.

        Params:
            jsfs: joint-sfs
            pops: populations to project onto.

        Returns:
            A projected jsfs.
        """
        mask = jnp.zeros(self.d, dtype=bool)
        mask = jax.ops.index_update(mask, jnp.array(pops), True)
        return self._replace(
            sample_sizes=jnp.array(self.sample_sizes)[mask], sites=self.sites[:, mask]
        )

    def downsample(self, new_sample_sizes: list[int | None]) -> "JSFS":
        """Downsamples the jsfs to the given sample size.

        Params:
            jsfs: joint-sfs
            new_sample_sizes: new sample sizes, or None if no change.

        Returns:
            A downsampled jsfs.

        Notes:
            This reduces the sparsity of the jsfs.
        """
        ret = self.to_COO()
        for ind, (n, m) in enumerate(zip(self.sample_sizes, new_sample_sizes)):
            if m is None:
                continue
            j = np.arange(m + 1)[None, :]
            i = np.arange(n + 1)[:, None]
            H = scipy.stats.hypergeom(n, i, m).pmf(j)
            H = sparse.COO.from_numpy(H)
            ret = sparse.moveaxis(
                sparse.tensordot(ret, H, axes=(ind, 0), return_type=sparse.COO), -1, ind
            )
        return JSFS.from_COO(ret)
