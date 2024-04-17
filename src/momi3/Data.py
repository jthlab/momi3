from functools import lru_cache
from math import ceil
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from sparse import COO

from momi3.Params import get_body
from momi3.utils import one_hot, ones


class Data(NamedTuple):
    """Data class.

    Attributes::
        sfs (jnp.ndarray): Observed site frequency spectrum vector.
            sfs_i is the observed site frequency for num_deriveds[pop][i]
        num_deriveds (dict[str, jnp.ndarray]): sfs entry configirations
            num_deriveds[pop][i] = X_{pop} number of deriveds in pop
        n_samples (dict[str, int]): Sample sizes.
        freqs_matrix (jnp.ndarray): Sparse matrix representing the frequencies
            at each locus. The (i,j)-th entry gives the frequency of the i-th
            config in (num_deriveds[pop_0][j], ..., num_deriveds[pop_k][j])
    """

    sfs_batches: list[np.ndarray]
    X_batches: list[dict[str, np.ndarray]]
    n_samples: dict[str, int]
    n_entries: int
    freqs_matrix: jnp.ndarray = None

    def _get_repr(self):
        n_possible_entries = jnp.prod(
            jnp.array(list([i + 1 for i in self.n_samples.values()]))
        )
        n_possible_entries -= 2  # discard all ancestral and all deriveds
        n_entries = self.n_entries
        if self.freqs_matrix is not None:
            num_loci = self.freqs_matrix.shape[1]
        else:
            num_loci = "freqs_matrix is not set"
        names = [
            "# of sfs entries",
            "# of possible sfs entries",
            "% of non zero entries",
            "# loci",
        ]
        density = 100 * n_entries / n_possible_entries
        if density == int(density):
            density = str(int(density))
        else:
            density = "{:.2g}%".format(density)
        values = [f"{n_entries}", f"{n_possible_entries}", density, num_loci]

        return names, values

    # def _jsfs(self):
    #     return sparse.COO(
    #         coords=tuple(self.num_deriveds[i] for i in self.n_samples),
    #         data=self.sfs,
    #         shape=tuple(self.n_samples[i] + 1 for i in self.n_samples))

    def __repr__(self):
        names, values = self._get_repr()

        out = []
        for n, v in zip(names, values):
            out.append(f"{names}: {values}")
        return "\n".join(out)

    def _repr_html_(self):
        names, values = self._get_repr()

        rows = zip(names, values)
        body = get_body(
            rows,
            styles=[
                "text-align:left; font-family:'Lucida Console', monospace",
                "text-align:right; font-family:'Lucida Console', monospace",
            ],
        )

        return f"""
<html>
<head>
<style>

.row {{
  margin-left:-5px;
  margin-right:-5px;
}}
table {{
  border-collapse: collapse;
  border-spacing: 0;
  width: 100%;
  border: 1px solid #ddd;
}}
</style>
</head>
<body>
    <div class="row">
        <div style="width:50%" class="column1">
            <table border="1" class="dataframe">
                <tbody>
                {body}
                </tbody>
            </table>
            </div>
        </div>
    </body>
</html>
"""


def get_data(
    sampled_demes: tuple[str],
    sample_sizes: tuple[int],
    leaves: set[str],
    jsfs: Union[COO, jnp.ndarray, np.ndarray],
    batch_size: int,
):
    n_samples = dict(zip(sampled_demes, sample_sizes))
    n_jsfs = [i - 1 for i in jsfs.shape]

    assert all(
        [i == j for i, j in zip(n_jsfs, sample_sizes)]
    ), f"dimensions of jsfs({jsfs.shape}) != sample sizes({sample_sizes})"

    non_zero_indices = [tuple(a.tolist()) for a in jsfs.nonzero()]
    index_tuples = list(zip(*non_zero_indices))
    # remove all ancestral and all deriveds, if they are present.
    for k in [(0,) * len(n_jsfs), tuple(n_jsfs)]:
        if k in index_tuples:
            index_tuples.remove(k)
    # then transpose back to list of tuples
    non_zero_indices = tuple(zip(*index_tuples))

    if isinstance(jsfs, COO):
        sfs = jsfs[non_zero_indices].todense()
    elif isinstance(jsfs, jnp.ndarray):
        sfs = jsfs[non_zero_indices]
    elif isinstance(jsfs, np.ndarray):
        sfs = jsfs[non_zero_indices]
    else:
        raise TypeError("Supported Types: sparse.COO, jax.numpy.ndarray, numpy.ndarray")

    sfs = np.array(sfs, dtype="f")
    num_deriveds = dict(zip(sampled_demes, non_zero_indices))

    n_entries = len(sfs)
    sfs_batches = get_sfs_batches(sfs, batch_size)
    deriveds = tuple([tuple(num_deriveds[pop]) for pop in sampled_demes])
    X_batches = get_X_batches(
        sampled_demes,
        sample_sizes,
        tuple(leaves),
        deriveds,
        batch_size,
        add_etbl_vecs=True,
    )

    return Data(sfs_batches, X_batches, n_samples, n_entries, freqs_matrix=None)


def get_sfs_batches(sfs, batch_size=None):
    n_devices = jax.device_count()
    len_sfs = len(sfs)
    n_entries = len_sfs + n_devices * 3

    n_for_device = ceil(n_entries / n_devices)
    if batch_size is None:
        batch_size = n_for_device
    else:
        pass

    n_for_device = ceil(n_entries / n_devices)
    n_for_map = ceil(n_for_device / batch_size)
    n_for_vmap = batch_size
    n_surplus = n_for_map * n_for_vmap * n_devices - len_sfs - n_devices * 3
    sfs = np.pad(sfs, [0, n_surplus])
    return sfs.reshape(n_devices, -1)


@lru_cache(None)
def get_X_batches(
    sampled_demes, sample_sizes, leaves, deriveds, batch_size=None, add_etbl_vecs=True
):
    n_devices = jax.device_count()
    n_entries = len(deriveds[0])
    if add_etbl_vecs:
        n_entries += n_devices * 3

    n_for_device = ceil(n_entries / n_devices)
    if batch_size is None:
        batch_size = n_for_device
    else:
        pass

    # batch_size = min(batch_size, n_for_device)
    n_for_map = ceil(n_for_device / batch_size)
    n_for_vmap = batch_size  # ceil(n_entries / n_for_map / n_devices)

    num_deriveds = {}
    for pop, derived in zip(sampled_demes, deriveds):
        num_deriveds[pop] = iter(derived)

    n_samples = dict(zip(sampled_demes, sample_sizes))

    device_load = n_for_map * n_for_vmap

    X = {}
    for pop in leaves:
        n = n_samples.get(pop, 0)
        X[pop] = []
        for dev_i in range(n_devices):
            if add_etbl_vecs:
                # In each device first three are for Total Branch Length
                X[pop].append(ones(n + 1))
                X[pop].append(one_hot(n + 1, 0))
                X[pop].append(one_hot(n + 1, n))
                next_loop = device_load - 3
            else:
                next_loop = device_load

            for _ in range(next_loop):
                if n == 0:
                    X[pop].append([1.0])
                else:
                    # Rest for entries
                    d = next(num_deriveds[pop], 0)
                    X[pop].append(one_hot(n + 1, d))

        X[pop] = (
            np.array(X[pop]).astype(float).reshape(n_devices, n_for_map, n_for_vmap, -1)
        )

    return X
