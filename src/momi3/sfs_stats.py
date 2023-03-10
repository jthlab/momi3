# flake8: noqa: E741
import collections as co
import functools as ft

import jax.numpy as jnp
import pandas as pd
import seaborn
from cached_property import cached_property
from matplotlib import pyplot as plt
from scipy.special import comb


def count_subsets(data, derived_weights_dict):
    total_counts_dict = dict(zip(data.sampled_pops, data.sample_size))
    ret = jnp.ones_like(data.sfs)
    for p, n in total_counts_dict.items():
        i = data.sampled_pops.index(p)
        curr = jnp.zeros_like(data.sfs)

        for d, w in enumerate(derived_weights_dict[(p,)]):
            if w == 0:
                continue
            a = n - d
            curr += w * (
                comb(total_counts_dict[p] - data.mutant_sizes[:, i], a)
                * comb(data.mutant_sizes[:, i], d)
            )
        ret *= curr

    return ret


class SfsStats(object):
    def __init__(self, momi, sampled_n_dict):
        self.sampled_n_dict = {p: n for p, n in sampled_n_dict.items() if n > 0}
        self.momi = momi

    def tensor_prod(self, derived_weights_dict):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    @cached_property
    def denom(self):
        return self.denom()

    def ordered_prob(self, subsample_dict, fold=False):
        sampled_n_dict = self.momi.data.sampled_n_dict
        subsample_dict = subsample_dict.copy()
        if fold:
            rev_subsample = {p: 1 - jnp.array(s) for p, s in subsample_dict.items()}

            return self.ordered_prob(subsample_dict.copy()) + self.ordered_prob(
                rev_subsample.copy()
            )

        derived_weights_dict = {}
        pops = [pop for pop, pop_subsample in subsample_dict.items()]
        pop_subsamples = [
            pop_subsample for pop, pop_subsample in subsample_dict.items()
        ]
        for pop, pop_subsample in zip(pops, pop_subsamples):
            n = sampled_n_dict[pop]
            arange = jnp.arange(n + 1)
            cnts = co.Counter([int(i) for i in pop_subsample])

            prob = jnp.ones(n + 1)
            for i in range(cnts[0]):
                prob *= n - arange - i
            for i in range(cnts[1]):
                prob *= arange - i
            for i in range(cnts[0] + cnts[1]):
                prob /= float(n - i)

            derived_weights_dict[(pop,)] = prob

        for pop in sampled_n_dict:
            if (pop,) in derived_weights_dict:
                pass
            else:
                n = sampled_n_dict[pop]
                derived_weights_dict[(pop,)] = jnp.ones(n + 1)
        # print(20*'***')
        return self.tensor_prod(derived_weights_dict) / self.denom

    def count_1100(self, A, B, C, O=None):
        # O=None -> O is the root population
        subsample_dict = co.defaultdict(list)
        subsample_dict[A].append(1)
        subsample_dict[B].append(1)
        subsample_dict[C].append(0)
        if O is not None:
            subsample_dict[O].append(0)

        return self.ordered_prob(subsample_dict, fold=(O is not None))

    def baba(self, A, B, C, *O):
        return self.count_1100(A, C, B, *O)

    def abba(self, A, B, C, *O):
        return self.count_1100(B, C, A, *O)

    def abba_baba(self, A, B, C, D=None):
        """
        Same as :meth:`f4`
        """
        return self.baba(A, B, C, D) - self.abba(A, B, C, D)

    def log_abba_baba(self, A, B, C, D=None):
        """
        Returns log(BABA/ABBA) = log(BABA)-log(ABBA)
        """
        return self.log(self.baba(A, B, C, D)) - self.log(self.abba(A, B, C, D))

    def f_st(self, A, B):
        """
        Returns (pi_between - pi_within) / pi_between, \
        where pi_between, pi_within represent the average number of pairwise \
        diffs between 2 individuals sampled from different or the same \
        population, respectively.
        """
        pi_within = (
            self.ordered_prob({A: [1, 0]}, fold=True)
            + self.ordered_prob({B: [1, 0]}, fold=True)
        ) / 2
        pi_between = self.ordered_prob({A: [1], B: [0]}, fold=True)

        return (pi_between - pi_within) / pi_between

    def f4(self, A, B, C, D=None):
        """
        Returns the ABBA-BABA (f4) statistic for testing admixture.

        :param str A: First population
        :param str B: Second population
        :param str C: Third population
        :param str D: Fourth population. If None, use ancestral allele.
        """
        return self.abba_baba(A, B, C, D)

    def f3(self, A, B, O):
        """
        Computes f3 statistic (O-A)*(O-B)

        :param str A: First population
        :param str B: Second population
        :param str O: Third population.
        """
        return self.f4(O, A, O, B)

    def f2(self, A, B):
        """
        Computes f2 statistic (A-B)*(A-B)

        :param str A: First population
        :param str B: Second population
        """
        return self.f4(A, B, A, B)

    def pattersons_d(self, A, B, C, D=None):
        """
        Returns Patterson's D, defined as (BABA-ABBA)/(BABA+ABBA).

        :param str A: First population
        :param str B: Second population
        :param str C: Third population
        :param str D: Fourth population. If None, use ancestral allele.
        """
        abba = self.abba(A, B, C, D)
        baba = self.baba(A, B, C, D)
        return (baba - abba) / (baba + abba)

    def greens_f(self, A, B, C, *O):
        # Estimate for the admixture of C into B in tree (((A,B),C),O)
        return self.abba_baba(A, B, C, *O) / self.abba_baba(A, C, C, *O)

    def f4_ratio(self, A, B, C, X, *O):
        # For tree (((A,B),C),O), and X admixed between B,C
        # an estimate for the admixture proportion from B
        # ref: Patterson et al 2012, Ancient Admixture in Human History, eq (4)
        return self.f4(X, C, A, *O) / self.f4(B, C, A, *O)

    def singleton_probs(self, pops):
        denom = None
        probs = {}
        for pop in pops:
            prob = self.ordered_prob(
                dict([(p, [1]) if p == pop else (p, [0]) for p in pops]), fold=True
            )
            probs[pop] = prob
            if denom is None:
                denom = prob
            else:
                denom = denom + prob
        return {"probs": probs, "denom": 1 - denom}


class SfsModelFitStats(SfsStats):
    """Class to compare expected vs. observed statistics of the SFS.

    All methods return :class:`JackknifeGoodnessFitStat` unless
    otherwise stated.

    Currently, all goodness-of-fit statistics are based on the multinomial
    SFS (i.e., the SFS normalized to be a probability distribution
    summing to 1). Thus the mutation rate has no effect on these statistics.

    See `Patterson et al 2012 <http://www.genetics.org/content/192/3/1065>`_,
    for definitions of f2, f3, f4 (abba-baba), and D statistics.

    Note this class does NOT get updated when the underlying
    ``demo_model`` changes; a new :class:`SfsModelFitStats` needs
    to be created to reflect any changes in the demography.

    :param momi.DemographicModel demo_model: Demography to compute expected \
    statistics under.

    :param dict sampled_n_dict: The number of samples to use \
    per population. SNPs with fewer than this number of samples \
    are ignored. The default is to use the \
    full sample size of the data, i.e. to remove all SNPs with any missing \
    data. For datasets with large amounts of missing data, \
    this could potentially lead to most or all SNPs being removed, so it is \
    important to specify a smaller sample size in such cases.
    """

    def __init__(self, momi):
        self.empirical = ObservedSfsStats(momi)
        self.expected = ExpectedSfsStats(momi)
        self.momi = momi

    def tensor_prod(self, derived_weights_dict):
        r"""Compute rank-1 tensor products of the SFS, which can be used \
        to express a wide range of SFS-based statistics.

        More specifically, this computes the sum

        .. math:: \sum_{i,j,\ldots} SFS_{i,j,\ldots} w^{(1)}_i w^{(2)}_j \cdots

        where :math:`w^{(1)}_i` is the weight corresponding to SFS entries \
        with ``i`` derived alleles in population 1, etc. Note the SFS is \
        normalized to sum to 1 here (it is a probability).

        :param dict derived_weights_dict: Maps leaf populations to \
        vectors (:class:`numpy.ndarray`). If a population has ``n`` samples \
        then the corresponding vector ``w`` should have length ``n+1``, \
        with ``w[i]`` being the weight for SFS entries with ``i`` copies of \
        the derived allele in the population.

        :rtype: :class:`JackknifeGoodnessFitStat`
        """
        exp = self.expected.tensor_prod(derived_weights_dict)
        emp = self.empirical.tensor_prod(derived_weights_dict)

        return JackknifeGoodnessFitStat(exp, emp.est, emp.jackknife)

    def log(self, x):
        return x.apply(jnp.log)

    @property
    def denom(self):
        return 1.0

    def all_pairs_ibs(self, fig=True):
        """Fit the IBS fraction for all pairs of populations, and optionally plot it.

        :param bool fig: whether to plot it
        :rtype: :class:`pandas.DataFrame`
        """
        pops = list(self.sampled_n_dict.keys())

        df = []
        for pop1 in pops:
            for pop2 in pops:
                if pop1 > pop2:
                    continue
                elif pop1 == pop2:
                    if self.sampled_n_dict[pop1] == 1:
                        continue
                    prob = self.ordered_prob({pop1: [0, 0]}, fold=True)
                else:
                    prob = self.ordered_prob({pop1: [0], pop2: [0]}, fold=True)

                line = [pop1, pop2, prob.expected, prob.observed, prob.z_score]
                df.append(line)

        return self._pairwise_zscores(df, fig)

    def all_f2(self, fig=True):
        pops = [k for k, v in self.sampled_n_dict.items() if v > 1]

        df = []
        for pop1 in pops:
            for pop2 in pops:
                if pop1 >= pop2:
                    continue
                else:
                    prob = self.f2(pop1, pop2)

                line = [pop1, pop2, prob.expected, prob.observed, prob.z_score]
                df.append(line)

        return self._pairwise_zscores(df, fig)

    def _pairwise_zscores(self, df, fig):
        ret = pd.DataFrame(
            sorted(df, key=lambda x: abs(x[-1]), reverse=True),
            columns=["Pop1", "Pop2", "Expected", "Observed", "Z"],
        )

        if fig:
            pivoted = ret.pivot(index="Pop1", columns="Pop2", values="Z")
            plt.gcf().clear()
            seaborn.heatmap(pivoted, annot=True, center=0)
            plt.title("Residual (Observed-Expected) Z-scores")
        return ret

    @property
    def n_subsets(self):
        return self.empirical.n_subsets

    @property
    def n_jackknife_blocks(self):
        return self.empirical.n_jackknife_blocks


class ObservedSfsStats(SfsStats):
    def __init__(self, momi, sampled_n_dict=None):
        # is_ascertained = dict(zip(sfs.sampled_pops, sfs.ascertainment_pop))
        # if sum(n for p, n in sampled_n_dict.items()
        #       if is_ascertained[p]) < 2:
        #    raise ValueError("sampled_n_dict must contain at least 2 ascertained alleles")
        self.momi = momi
        sampled_n_dict = momi.data.sampled_n_dict
        super(ObservedSfsStats, self).__init__(momi, sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        data = self.momi.data
        weighted_counts = count_subsets(data, derived_weights_dict)

        # subtract out weights of monomorphic
        mono_anc = {}
        mono_der = {}
        for pop in data.sampled_pops:
            try:
                v = derived_weights_dict[pop]
            except KeyError:
                try:
                    v = [1] * (data.sampled_n_dict[pop] + 1)
                except KeyError:
                    continue
            if True:
                mono_anc[(pop,)] = [v[0]] + [0] * (len(v) - 1)
                mono_der[(pop,)] = [0] * (len(v) - 1) + [v[-1]]
            else:
                mono_anc[(pop,)] = v
                mono_der[(pop,)] = v
        mono_anc = count_subsets(data, mono_anc)
        mono_der = count_subsets(data, mono_der)

        return JackknifeStat.from_chunks(
            self.momi.data.freqs_matrix.T.dot(weighted_counts - mono_anc - mono_der)
        )

    def log(self, x):
        return x.apply(jnp.log)

    @property
    def n_subsets(self):
        return self.denom.est

    @property
    def n_jackknife_blocks(self):
        return self.sfs.n_loci


class ExpectedSfsStats(SfsStats):
    def __init__(self, momi):
        self.momi = momi
        self.data = self.momi.data
        sampled_n_dict = self.data.sampled_n_dict
        super(ExpectedSfsStats, self).__init__(momi, sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        momi = self.momi
        sampled_n_dict = self.data.sampled_n_dict
        ns = momi.get_lineage_size(sampled_n_dict)
        lineage_size = tuple(ns[i] for i in momi.graph_pops)
        phis = {event: 0.0 for event in momi.T.nodes()}
        esfs = momi.EvaluateEventTree(
            derived_weights_dict, phis, momi.prms, lineage_size
        )
        etbl = momi._etbl(momi.prms, lineage_size)
        return esfs / etbl

    def log(self, x):
        return jnp.log(x)

    @cached_property
    def denom(self):
        momi = self.momi
        sampled_n_dict = self.data.sampled_n_dict
        ns = momi.get_lineage_size(sampled_n_dict)
        lineage_size = tuple(ns[i] for i in momi.graph_pops)
        etbl = momi._etbl(momi.prms, lineage_size)
        return etbl


class JackknifeGoodnessFitStat(object):
    """
    Object returned by methods of :class:`SfsModelFitStats`.

    Basic arithmetic operations are supported, allowing to build
    up complex statistics out of simpler ones.

    The raw expected, observed, and jackknifed_array values
    can be accessed as attributes of this class.

    :param float expected: the expected value of the statistic
    :param float observed: the observed value of the statistic
    :param numpy.ndarray jackknifed_array: array of the jackknifed \
    values of the statistic.
    """

    def __init__(self, expected, observed, jackknifed_array):
        self.expected = float(expected)
        self.observed = observed
        self.jackknifed_array = jackknifed_array

    @property
    def sd(self):
        """
        Standard deviation of the statistic, estimated via jackknife
        """
        resids = self.jackknifed_array - self.observed
        return jnp.sqrt(jnp.mean(resids**2) * (len(self.jackknifed_array) - 1))

    @property
    def bias(self):
        return jnp.mean(self.jackknifed_array - self.observed) * (
            len(self.jackknifed_array) - 1
        )

    @property
    def z_score(self):
        """
        Z-score of the statistic, defined as (observed-expected)/sd
        """
        return (self.observed - self.expected) / self.sd

    def __repr__(self):
        return (
            "JackknifeGoodnessFitStat(expected={}, observed={},"
            " bias={}, sd={}, z_score={})"
        ).format(self.expected, self.observed, self.bias, self.sd, self.z_score)

    def apply(self, fun):
        return JackknifeGoodnessFitStat(
            fun(self.expected), fun(self.observed), fun(self.jackknifed_array)
        )

    def __add__(self, other):
        other = self._get_other(other)
        return JackknifeGoodnessFitStat(
            self.expected + other.expected,
            self.observed + other.observed,
            self.jackknifed_array + other.jackknifed_array,
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (self * (-1)) + other

    def __mul__(self, other):
        other = self._get_other(other)
        return JackknifeGoodnessFitStat(
            self.expected * other.expected,
            self.observed * other.observed,
            self.jackknifed_array * other.jackknifed_array,
        )

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = self._get_other(other)
        return JackknifeGoodnessFitStat(
            self.expected**other.expected,
            self.observed**other.observed,
            self.jackknifed_array**other.jackknifed_array,
        )

    def __rpow__(self, other):
        other = self._get_other(other)
        return other**self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

    def _get_other(self, other):
        try:
            other.expected, other.observed, other.jackknifed_array
        except AttributeError:
            return type(self)(other, other, other)
        else:
            return other


def jackknife_arr_op(wrapped_op):
    @ft.wraps(wrapped_op)
    def wraps_op(self, other):
        try:
            other.est, other.jackknife
        except AttributeError:
            return wrapped_op(self, JackknifeStat(other, other))
        else:
            return wrapped_op(self, other)

    return wraps_op


class JackknifeStat(object):
    @classmethod
    def from_chunks(cls, x):
        tot = jnp.sum(x)
        return cls(tot, tot - x)

    def __init__(self, est, jackknife):
        self.est = est
        self.jackknife = jackknife

    def apply(self, fun):
        return JackknifeStat(fun(self.est), fun(self.jackknife))

    @jackknife_arr_op
    def __add__(self, other):
        return JackknifeStat(self.est + other.est, self.jackknife + other.jackknife)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return JackknifeStat(-self.est, -self.jackknife)

    @jackknife_arr_op
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @jackknife_arr_op
    def __mul__(self, other):
        return JackknifeStat(self.est * other.est, self.jackknife * other.jackknife)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

    @jackknife_arr_op
    def __pow__(self, other):
        return JackknifeStat(self.est**other.est, self.jackknife**other.jackknife)

    @jackknife_arr_op
    def __rpow__(self, other):
        return JackknifeStat(other.est**self.est, other.jackknife**self.jackknife)

    @property
    def resids(self):
        return self.jackknife - self.est

    @property
    def var(self):
        return jnp.mean(self.resids**2) * (len(self.jackknife) - 1)

    @property
    def bias(self):
        return jnp.mean(self.resids) * (len(self.jackknife) - 1)

    @property
    def sd(self):
        return jnp.sqrt(self.var)

    @property
    def z_score(self):
        return self.est / self.sd

    def __repr__(self):
        return "JackknifeStat(est={}, sd={}) at {}".format(
            self.est, self.sd, hex(id(self))
        )
