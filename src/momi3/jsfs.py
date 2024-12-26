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
        sites: The coordinates of the nonzero entries.
        counts: The counts of the nonzero entries.

    Notes:
        The JSFS is a sparse tensor, where the dimensions are the sample sizes of the
        populations (plus one). The nonzero entries are the counts of the number of
        sites with a given frequency of derived alleles.
    """

    sample_sizes: dict[str, int]
    sites: Int[Array, "s d"]  # noqa: F722
    counts: Int[Array, "s"]

    def to_COO(self) -> sparse.COO:
        return sparse.COO(
            self.sites.T, self.counts, shape=tuple(n + 1 for n in self.ns)
        )

    @classmethod
    def from_dense(
        cls, dense: Array, pops: list[str], drop_nonseg: bool = True
    ) -> "JSFS":
        ret = cls(
            sample_sizes=dict(zip(pops, [s - 1 for s in dense.shape])),
            sites=jnp.array(list(np.ndindex(*dense.shape))),
            counts=dense.ravel(),
        )
        if drop_nonseg:
            ret = ret.drop_nonseg_sites()
        return ret

    @classmethod
    def from_COO(cls, coo: sparse.COO, pops: list[str]) -> "JSFS":
        return cls(
            sample_sizes=dict(zip(pops, [s - 1 for s in coo.shape])),
            sites=coo.coords.T,
            counts=coo.data,
        )

    def todense(self) -> Array:
        return self.to_COO().todense()

    @property
    def pops(self):
        return list(self.sample_sizes.keys())

    @property
    def ns(self):
        return list(self.sample_sizes.values())

    @property
    def s(self) -> int:
        return len(self.counts)

    @property
    def d(self) -> int:
        return len(self.sample_sizes)

    @property
    def num_seg_sites(self):
        return (self.counts * (~self.nonseg_sites)).sum()

    def drop_nonseg_sites(self) -> "JSFS":
        """Return a jsfs with non-segregating sites removed."""
        return self._replace(
            sites=self.sites[~self.nonseg_sites], counts=self.counts[~self.nonseg_sites]
        )

    def slice(self, i: int) -> list["JSFS"]:
        "Slice jsfs into i sub-jsfs of approximately equal size."
        return [
            self._replace(sites=self.sites[j], counts=self.counts[j])
            for j in np.array_split(np.arange(self.s), i)
        ]

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
        s1 = jnp.all(self.sites == 0, axis=1)
        s2 = jnp.all(self.sites == jnp.array(self.ns), axis=1)
        return s1 | s2

    def project(self, pops: list[str]) -> "JSFS":
        """Projects the jsfs to a subset of populations.

        Params:
            jsfs: joint-sfs
            pops: populations to keep.

        Returns:
            A projected jsfs.
        """
        ind = [self.pops.index(k) for k in pops]
        sites = self.sites[:, ind]
        H, br = np.histogramdd(
            sites,
            bins=[np.arange(self.sample_sizes[p] + 2) for p in pops],
            weights=self.counts,
        )
        ret = JSFS.from_COO(sparse.COO.from_numpy(H), pops)
        ns = ret.nonseg_sites
        return ret._replace(sites=ret.sites[~ns], counts=ret.counts[~ns])

    def downsample(self, new_sample_sizes: dict[str, int]) -> "JSFS":
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
        for ind, (k, n) in enumerate(self.sample_sizes.items()):
            m = new_sample_sizes.get(k, n)
            if m > n:
                raise ValueError(f"Cannot upsample population {k}.")
            if m == n:
                # no change
                continue
            j = np.arange(m + 1)[None, :]
            i = np.arange(n + 1)[:, None]
            H = scipy.stats.hypergeom(n, i, m).pmf(j)
            H = sparse.COO.from_numpy(H)
            ret = sparse.moveaxis(
                sparse.tensordot(ret, H, axes=(ind, 0), return_type=sparse.COO), -1, ind
            )
        ret = JSFS.from_COO(ret, self.pops)
        ns = ret.nonseg_sites
        # remove non-segregating sites
        return ret._replace(sites=ret.sites[~ns], counts=ret.counts[~ns])
