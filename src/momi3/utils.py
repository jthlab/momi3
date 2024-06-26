import os
import platform
import re
import subprocess
import timeit
from functools import lru_cache, partial
from typing import Callable, Union

import demes
import jax
import jax.numpy as jnp
import msprime as msp
import numpy as np
import scipy
import sparse
import tskit
from gmpy2 import mpq
from jax.numpy import diag, dot, exp, log
from jax.scipy.special import logsumexp
from jax.tree_util import register_pytree_node
from joblib import Parallel, delayed
from sparse._coo.core import COO
from tqdm.autonotebook import tqdm

from .math_functions import expm1d, log_hypergeom

register_pytree_node(
    demes.Graph,
    lambda g: ((), demes.dumps(g, simplified=False)),
    lambda aux_data, _: demes.loads(aux_data),
)


@lru_cache(None)
def admix_log_hyp(nw, nv1, nv2):  # , expon = False):
    # used by
    x1, x2, xw, m1, j1 = jnp.ogrid[
        tuple(slice(None, lim + 1) for lim in (nv1, nv2, nw, nw, nw))
    ]
    m2 = nw - m1
    j2 = xw - j1
    log_hyps = log_hypergeom(n=x1, k=j1, M=nv1, N=m1) + log_hypergeom(
        n=x2, k=j2, M=nv2, N=m2
    )
    ret = logsumexp(log_hyps, axis=-1)  # [x1, x2, xw, m1]
    return ret


#     if expon:
#         return exp(ret)
#     else:
#         return ret


@lru_cache(None)
def admix_hyp(nw, nv1, nv2):
    return jnp.exp(admix_log_hyp(nw, nv1, nv2))


@lru_cache(None)
def W_matrix(n: int) -> np.ndarray:
    # n should be castable to int. otherwise mpq() constructor will fail.
    n = int(n)
    # returns W matrix as calculated as eq 13:15 @ Polanski 2013
    # n: sample size
    if n == 1:
        return np.array([[]])
    W = np.zeros(
        [n - 1, n - 1], dtype=object
    )  # indices are [b, j] offset by 1 and 2 respectively
    W[:, 2 - 2] = mpq(6, n + 1)
    if n == 2:
        return W
    b = list(range(1, n))
    W[:, 3 - 2] = np.array([mpq(30 * (n - 2 * bb), (n + 1) * (n + 2)) for bb in b])
    for j in range(2, n - 1):
        A = mpq(-(1 + j) * (3 + 2 * j) * (n - j), j * (2 * j - 1) * (n + j + 1))
        B = np.array([mpq((3 + 2 * j) * (n - 2 * bb), j * (n + j + 1)) for bb in b])
        W[:, j + 2 - 2] = A * W[:, j - 2] + B * W[:, j + 1 - 2]
    return W


@lru_cache(None)
def rate_matrix(n, sparse_format="csr"):
    # returns moran rate matrix (from momi2)
    # n: sample size
    i = np.arange(n + 1)
    diag = i * (n - i) / 2.0
    diags = [diag[:-1], -2 * diag, diag[1:]]
    M = scipy.sparse.diags(diags, [1, 0, -1], (n + 1, n + 1), format=sparse_format)
    return M.T


@lru_cache(None)
def moran_eigensystem(n):
    # returns eigen decomp of rate matrix (from momi2)
    # n: sample size
    M = rate_matrix(n).toarray()
    d, Q = np.linalg.eig(M)
    return d, Q.T


@partial(jax.jit)
def moran_transition(t, d, Q):
    """
    returns transition rate at given trancation time
    tau: truncation time
    n: sample size
    """
    D = diag(exp(t * d))
    return jnp.linalg.solve(Q, dot(D, Q))


@jax.jit
def growth_rate(Ne0, Ne1, tau):
    return log(Ne0 / Ne1) / tau


@jax.jit
def pop_size_at_t(Ne0, b, t):
    return Ne0 * exp(-b * t)


@jax.jit
def RateEGPS(Ne0, b, tau):
    tau0 = jnp.isclose(tau, 0.0)
    tau_safe = jnp.where(tau0, 1.0, tau)
    ret = jnp.where(tau0, 0.0, expm1d(tau_safe * b) * tau_safe / Ne0)
    return ret


@partial(jax.jit)
def RateCPS(Ne0, tau):
    return tau / Ne0


@partial(jax.jit, static_argnums=(0,))
def sample_m(n, p, seed):
    """
    Sample from a binomial distribution and returns the sampled value and the new seed.

    Parameters
    ----------
    n : int
        sample size
    p : float
        success probability
    seed : int
        random seed

    Returns
    -------
    ret : int
        Drawn samples from the binomial(n, p)
    seed : int
        The newly generated seed
    """
    seed_use, seed = jax.random.split(jax.random.PRNGKey(seed), 1)[0]  # split the seed
    ret = jax.random.bernoulli(jax.random.PRNGKey(seed_use), p=p, shape=(n,)).sum()
    return ret.astype(int), seed


def exp_integralEGPS_quad(a, b, tau, j, error=False):
    from scipy.integrate import quad

    def f(t):
        return exp(-j * a * (exp(b * t) - 1) / b)

    ret = quad(f, 0, tau)
    if error:
        return ret
    else:
        return ret[0]


def msprime_chromosome_simulator(
    demo: demes.Graph,
    sampled_demes: tuple[str],
    sample_sizes: tuple[int],
    sequence_length: int,
    recombination_rate: float,
    mutation_rate: float,
    seed: int = None,
    low_memory: bool = True,
) -> Union[sparse.COO, np.ndarray]:
    chr_sim = msp.sim_ancestry(
        ploidy=1,
        demography=msp.Demography.from_demes(demo),
        samples=dict(zip(sampled_demes, sample_sizes)),
        recombination_rate=recombination_rate,
        sequence_length=int(sequence_length),
        random_seed=seed,
    )
    mt_chr = msp.sim_mutations(chr_sim, rate=mutation_rate, random_seed=seed)

    sample_ids = [
        [
            list(mt_chr.samples(pop.id))
            for pop in mt_chr.populations()
            if pop.metadata["name"] == deme
        ][0]
        for deme in sampled_demes
    ]
    if low_memory:
        jsfs = tskit_low_memory_afs(mt_chr, sample_ids)
    else:
        jsfs = mt_chr.allele_frequency_spectrum(
            sample_sets=sample_ids, polarised=True, span_normalise=False
        )
    return jsfs


def msprime_simulator(
    demo: demes.Graph,
    sampled_demes: tuple[str],
    sample_sizes: tuple[int],
    num_replicates: int,
    seed: int = None,
) -> sparse.COO:
    # msprime sampler. Returns sparse Joint SFS array
    # TODO: Add parellel mode
    demo = msp.Demography.from_demes(demo)
    ancestry_reps = msp.sim_ancestry(
        ploidy=1,
        demography=demo,
        samples=dict(zip(sampled_demes, sample_sizes)),
        num_replicates=num_replicates,
        random_seed=seed,
    )

    ancestry_reps = list(ancestry_reps)
    tbl = np.mean([ts.first().total_branch_length for ts in ancestry_reps])
    rate = 1 / tbl / 5  # A small rate for mutations

    x = ancestry_reps[0]
    pop_ids = {
        deme: [pop.id for pop in x.populations() if pop.metadata["name"] == deme][0]
        for deme in sampled_demes
    }
    sample_ids = {deme: x.samples(pop_ids[deme]) for deme in sampled_demes}

    num_muts = []
    sparse_sfs = {}

    seeds = np.random.RandomState(seed).randint(2**31 - 1, size=num_replicates)

    for ts, seed in zip(tqdm(ancestry_reps), seeds):
        mts = msp.sim_mutations(ts, rate=rate, random_seed=seed)
        num_muts.append(mts.num_mutations)
        tree = mts.first()  # It has a single tree
        for mut in tree.mutations():  # loop over mutations
            config = []
            x = set(tree.leaves(mut.node))  # set of leaves that has the mutation
            for deme in sampled_demes:
                nmut = len(
                    x.intersection(sample_ids[deme])
                )  # size of intersection between deme and mutants
                config.append(nmut)
            config = tuple(config)

            try:
                sparse_sfs[config] += 1
            except Exception:
                sparse_sfs[config] = 1

    sparse_sfs = sparse.COO(sparse_sfs, shape=[i + 1 for i in sample_sizes])
    return sparse_sfs


def get_CPU_name() -> str:
    # Returns model and make of the cpu
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def get_GPU_name() -> str:
    # Returns model and make of the GPU if any
    device = jax.devices()[0]
    if device.platform == "gpu":
        GPU = device.device_kind
    else:
        GPU = None
    return GPU


def tskit_low_memory_afs(
    ts: tskit.trees.TreeSequence, sample_ids: list[list]
) -> sparse.COO:
    # Low memory calculation of Joint SFS array from tskit.trees.TreeSequence
    P = len(sample_ids)
    anc = {i.position: i.ancestral_state for i in ts.sites()}
    sites_allele_counter = {i: P * [0] for i in anc}

    sample_sizes = [len(sample_id) for sample_id in sample_ids]
    for p in range(P):
        n = sample_sizes[p]
        cur_ts = ts.simplify(sample_ids[p])
        varis = list(cur_ts.variants())
        for i in varis:
            pos = i.position
            anc_var = anc[pos]
            sites_allele_counter[pos][p] = n - i.counts()[anc_var]

    sites_allele_counter = list(sites_allele_counter.values())
    AFS = {}
    tt = sum(sample_sizes)
    for i in sites_allele_counter:
        s_i = sum(i)
        if (s_i == 0) | (s_i == tt):
            pass
        else:
            i = tuple(i)
            try:
                AFS[i] += 1
            except Exception:
                AFS[i] = 1

    mutant_sizes = jnp.array(list(AFS.keys()))
    sfs = jnp.array(list(AFS.values()))

    return sparse.COO(
        data=sfs, coords=mutant_sizes.T, shape=[i + 1 for i in sample_sizes]
    )


def update(p0: dict, path: tuple, val: float) -> dict:
    # Update the value of the key in the nested dictionary with the given path.
    current = p0
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = val
    return p0


def Parallel_runtime(f: Callable, num_replicates: int, n_jobs: int) -> list:
    # Run f parrallel using joblib.
    num_replicates = num_replicates
    return Parallel(n_jobs=n_jobs)(
        delayed(lambda: timeit.timeit(f, number=1))() for _ in range(num_replicates)
    )


def sum_dicts(dicts: list[dict]) -> dict:
    # Take the sum of dictionaries with same keys
    result = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                # If the value is a dict, recursively sum its values
                result[key] = sum_dicts([result.get(key, {}), value])
            else:
                # Otherwise, add the value to the result
                result[key] = result.get(key, 0) + value
    return result


def halfsigmoid(x, scale=10):
    return 2 * (1 / (1 + np.exp(-x / scale)) - 0.5)


def signif(x, ptype):
    if ptype in ["eta", "tau"]:
        str(int(x))
    else:
        return f"{x:.2g}"  # 2 digit significance


def one_hot(n, b):
    ret = n * [0]
    ret[b] = 1
    return ret


def ones(n):
    return n * [1]


def downsample_jsfs(jsfs: np.ndarray | COO, down_sample_to: list[int]) -> COO:
    """Returns downsampled jsfs

    Parameters
    ----------
    jsfs : np.ndarray
        joint-sfs
    down_sample_to : list[int]
        new shape of the jsfs will be down_sample_to + 1.
        If you don't want to sample some dimensions give Nones.
    Returns
    -------
    np.ndarray
        downsampled jsfs
    """
    down_sample_from = tuple(i - 1 for i in jsfs.shape)
    for ind, (n, m) in enumerate(zip(down_sample_from, down_sample_to)):
        if m is None:
            continue
        j = np.arange(m + 1)[None, :]
        i = np.arange(n + 1)[:, None]
        H = scipy.stats.hypergeom(n, i, m).pmf(j)
        H = COO.from_numpy(H)
        jsfs = sparse.moveaxis(
            sparse.tensordot(jsfs, H, axes=(ind, 0), return_type=COO), -1, ind
        )
    return jsfs


def bootstrap_sample(
    jsfs: Union[COO, jnp.ndarray, np.ndarray], n_SNPs: int = None, seed=None
) -> COO:
    np.random.seed(seed)
    nmuts = int(round(jsfs.sum()))
    jsfs_nonzero = jsfs.nonzero()
    nonzeros = [tuple(i) for i in np.array(jsfs_nonzero).T]
    if isinstance(jsfs, COO):
        p = jsfs.data
    else:
        p = jsfs[jsfs_nonzero]
    p = p / nmuts
    if n_SNPs is not None:
        nmuts = n_SNPs
    new_inds = np.random.choice(range(len(nonzeros)), p=p, size=nmuts)
    sfs = {}
    for new_ind in new_inds:
        config = nonzeros[new_ind]
        sfs[config] = sfs.get(config, 0) + 1
    return COO(sfs, shape=jsfs.shape)
