from typing import NamedTuple, Union

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
    return get_data_by_jsfs(sampled_demes, sample_sizes, leaves, jsfs, batch_size)


def get_data_by_jsfs(
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

    non_zero_indeces = jsfs.nonzero()

    # Remove all deriveds and all ancestral
    # sfs[0, 0, ..., 0] = sfs[-1, -1, ..., -1] = 0
    remove_first = True
    remove_last = True
    for i, n in zip(non_zero_indeces, n_jsfs):
        remove_first &= i[0] == 0
        remove_last &= i[-1] == n

    new_non_zero_indeces = []
    for i in non_zero_indeces:
        j = list(i)
        if remove_first:
            j = j[1:]
        if remove_last:
            j = j[:-1]
        new_non_zero_indeces.append(tuple(j))
    non_zero_indeces = tuple(new_non_zero_indeces)

    if isinstance(jsfs, COO):
        sfs = jsfs[non_zero_indeces].todense()
    elif isinstance(jsfs, jnp.ndarray):
        sfs = jsfs[non_zero_indeces]
    elif isinstance(jsfs, np.ndarray):
        sfs = jsfs[non_zero_indeces]
    else:
        raise TypeError("Supported Types: sparse.COO, jax.numpy.ndarray, numpy.ndarray")

    sfs = np.array(sfs, dtype="f")
    num_deriveds = dict(zip(sampled_demes, non_zero_indeces))

    n_entries = len(sfs)
    batch_size = min(batch_size, n_entries)
    if n_entries % batch_size == 0:
        n_batches = n_entries // batch_size
    else:
        n_batches = n_entries // batch_size + 1

    X_batches = []
    sfs_batches = []

    start = 0
    for i in range(n_batches):
        end = start + batch_size
        X = {}
        for pop in leaves:
            X[pop] = []
            n = n_samples.get(pop, 0)
            if n == 0:
                X[pop] = np.array((batch_size + 3) * [[1]])
            else:
                # First three are for Total Branch Length
                X[pop].append(ones(n + 1))
                X[pop].append(one_hot(n + 1, 0))
                X[pop].append(one_hot(n + 1, n))

                # Rest for entries
                for d in num_deriveds[pop][start:end]:
                    X[pop].append(one_hot(n + 1, d))

                X[pop] = np.array(X[pop])

        X_batches.append(X)
        sfs_batches.append(sfs[start:end])
        start = end

    if end != n_entries:
        X_batches[-1] = X_batches[-1] | {
            key: np.pad(X_batches[-1][key], [[0, end - n_entries], [0, 0]])
            for key in num_deriveds
        }
        sfs_batches[-1] = np.pad(sfs_batches[-1], [0, end - n_entries])
    else:
        pass

    return Data(sfs_batches, X_batches, n_samples, n_entries, freqs_matrix=None)
