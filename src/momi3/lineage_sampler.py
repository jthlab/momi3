from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import scipy
import sparse
from jax.scipy.special import gammaln as gammaln

from momi3 import events
from momi3.common import traverse
from momi3.event_tree import ETBuilder
from momi3.Params import Params


def log_factorial(x):
    return gammaln(x + 1)


def log_binom(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def downsample_jsfs(jsfs, down_sample_to):
    down_sample_from = tuple(i - 1 for i in jsfs.shape)
    for ind, (n, m) in enumerate(zip(down_sample_from, down_sample_to)):
        j = np.arange(m + 1)[None, :]
        i = np.arange(n + 1)[:, None]
        H = scipy.stats.hypergeom(n, i, m).pmf(j)
        jsfs = sparse.moveaxis(
            sparse.tensordot(jsfs, H, axes=(ind, 0), return_type=np.ndarray), -1, ind
        )
    return jsfs


@partial(jax.jit, static_argnums=3)
def sample_lift_constant(Ne, t0, t1, n, seed):
    # Constant growth surviving lineages sampler
    # Given n lineages, simulate the number of lineages t1 - t0 generations later. Coal rate = 1 / Ne.
    tau = t1 - t0
    js = jnp.arange(n, 1, -1)
    rng = jax.random.PRNGKey(seed)
    Zi = jax.random.exponential(rng, (n - 1,))
    waiting_times = Zi * Ne / jnp.exp(log_binom(js, 2))
    waiting_times = jnp.cumsum(waiting_times)
    ret = n - jnp.searchsorted(waiting_times, tau)
    return ret


def exp_intercoal_sampler(Ne0, g, t, Z, k):
    # Used by sample_lift_exponential
    return jnp.log(jnp.exp(g * t) + (g * Ne0 * Z) / jnp.exp(log_binom(k, 2))) / g - t


def lift_exp_growth_loop(carry, k):
    # Used by sample_lift_exponential
    rng, t, n, Ne0, g, tau = carry
    rng, rng_input = jax.random.split(rng)
    Zi = jax.random.exponential(rng_input)
    tmp = exp_intercoal_sampler(Ne0, g, t, Zi, k)
    Tk = jnp.array((tmp, tau - t)).min()
    t += Tk
    return [rng, t, n, Ne0, g, tau], t


@partial(jax.jit, static_argnums=4)
def sample_lift_exponential(Ne0, Ne1, t0, t1, n, seed):
    # Exponential growth surviving lineages sampler
    # Given n lineages, simulate the number of lineages t1 - t0 generations later. Coal rate = R(Ne0, Ne1, t1, t0)
    tau = t1 - t0
    g = (jnp.log(Ne1) - jnp.log(Ne0)) / tau
    rng = jax.random.PRNGKey(seed)
    carry = [rng, 0.0, n, Ne0, g, tau]
    carry, waiting_times = jax.lax.scan(
        lift_exp_growth_loop, carry, xs=jnp.arange(n, 1, -1)
    )
    ret = n - jnp.searchsorted(waiting_times, tau)
    return ret


@jax.jit
def migration_transition(ks, mig_params, seed):
    ks = jnp.array(ks)

    coal_rate = jnp.exp(log_binom(ks, 2)) * mig_params["coal"]
    mig_rate = ks[:, None] * mig_params["mig"]

    cumsum_coal_rate = jnp.cumsum(coal_rate)
    cumsum_mig_rate = jnp.cumsum(mig_rate).reshape(mig_rate.shape)

    total_coal_rate = cumsum_coal_rate[-1]
    total_mig_rate = cumsum_mig_rate[-1, -1]

    total_rate = total_coal_rate + total_mig_rate

    rng = jax.random.PRNGKey(seed)
    rng, rng_input = jax.random.split(rng)
    Zi = jax.random.exponential(rng_input)  # rv for waiting time
    rng, rng_input = jax.random.split(rng)
    Ui = jax.random.uniform(
        rng_input
    )  # rv to decide the event is a coalescent or migration
    rng, rng_input = jax.random.split(rng)
    Uj = jax.random.uniform(rng_input)  # rv to decide which migration or coalescent

    waiting_time = Zi / total_rate

    is_coal = total_coal_rate / total_rate > Ui

    which_coal = jnp.searchsorted(cumsum_coal_rate / total_coal_rate, Uj)
    which_mig = jnp.searchsorted(cumsum_mig_rate.flatten() / total_mig_rate, Uj)
    which_mig_r = which_mig // mig_rate.shape[0]
    which_mig_c = which_mig - which_mig_r * mig_rate.shape[0]

    return waiting_time, is_coal, which_coal, (which_mig_r, which_mig_c)


def sample_migration_constant(mig_params, tau, ns, seed):
    ks = list(ns).copy()
    t = 0.0
    while True:
        seed += 1
        (
            waiting_time,
            is_coal,
            which_coal,
            (which_mig_r, which_mig_c),
        ) = migration_transition(ks, mig_params, seed)

        t += waiting_time
        if t > tau:
            break

        if is_coal:
            ks[which_coal] -= 1
        else:
            ks[which_mig_r] -= 1
            ks[which_mig_c] += 1

    return ks


def sample_migration(
    n: dict,
    ev: events.Lift,
    theta_train_sample: np.ndarray,
    params: Params,
    seed: int = None,
    quantile: float = 0.95,
):
    raise NotImplementedError("No support for Migration")
    seeds = np.random.RandomState(seed).randint(
        1, 2**31 - 1, theta_train_sample.shape[0]
    )
    train_keys = params._train_keys

    ret = []
    for theta_train, seed in zip(theta_train_sample, seeds):
        for key, val in zip(train_keys, theta_train):
            params[key].set(val)
        demo_dict = params._demo_dict

        Ne_dict = ev._f_Ne(demo_dict)
        try:
            coal = jnp.array(
                [1 / Ne_dict[pop] for pop in Ne_dict]
            )  # only handles constant pop size migration
        except TypeError as e:
            raise NotImplementedError(
                "Exponential Growth Migration Sampler is not implemented"
            ) from e

        n0 = [n.get(i, 1) for i in Ne_dict]

        migs = ev._migmat(demo_dict, ev.migrations)

        mig = np.zeros(2 * [len(coal)])
        for i, pop1 in enumerate(Ne_dict):
            for j, pop2 in enumerate(Ne_dict):
                mig[i, j] = migs.get((pop1, pop2), 0.0)

        mig_params = {"coal": coal, "mig": mig}

        t0 = traverse(demo_dict, ev.t0.path)
        t1 = traverse(demo_dict, ev.t1.path)
        tau = t1 - t0

        ret.append(sample_migration_constant(mig_params, tau, n0, seed))

    n1 = {
        key: int(round(val))
        for key, val in zip(Ne_dict, np.quantile(ret, quantile, axis=0))
    }

    return n1


def sample_lift(
    n: dict,
    ev: events.Lift,
    theta_train_sample: np.ndarray,
    params: Params,
    seed: int = None,
    quantile: float = 0.95,
):
    """Samples surviving lineages and returns its user given quantile

    Args:
        n (int): sample sizes
        ev (Lift): Lift event
        theta_train_sample (np.ndarray): nrow: number of samples, ncol: number of training parameters
        params (Params): Parameters
        seed (int, optional): random seed
        quantile (float, optional): F(X < x) >= quantile
    """
    n1 = n.copy()

    demo_dict = params._demo_dict
    pops = list(ev._f_Ne(demo_dict).keys())
    seeds = np.random.RandomState(seed).randint(
        1, 2**31 - 1, theta_train_sample.shape[0]
    )

    t0s = []
    t1s = []
    Ne0s_dict = {pop: [] for pop in pops}
    Ne1s_dict = {pop: [] for pop in pops}

    train_keys = params._train_keys
    for theta_train in theta_train_sample:
        for key, val in zip(train_keys, theta_train):
            params[key].set(val)
        demo_dict = params._demo_dict

        t0s.append(traverse(demo_dict, ev.t0.path))
        t1s.append(traverse(demo_dict, ev.t1.path))

        sizes = ev._f_Ne(demo_dict)
        for pop in pops:
            s = sizes[pop]
            if isinstance(s, tuple):
                Ne0, Ne1 = s
            else:
                Ne0 = Ne1 = s

            Ne0s_dict[pop].append(Ne0)
            Ne1s_dict[pop].append(Ne1)

    t0s = jnp.array(t0s)
    t1s = jnp.array(t1s)
    for pop in pops:
        Ne0s = jnp.array(Ne0s_dict[pop])
        Ne1s = jnp.array(Ne1s_dict[pop])

        n0 = n.get(pop, 1)

        if jnp.isclose(Ne0s[0], Ne1s[1]):
            f = jax.vmap(sample_lift_constant, (0, 0, 0, None, 0))
            ret = f(Ne0s, t0s, t1s, n0, seeds)
        else:
            f = jax.vmap(sample_lift_exponential, (0, 0, 0, 0, None, 0))
            ret = f(Ne0s, Ne1s, t0s, t1s, n0, seeds)

        n1[pop] = int(round(np.quantile(ret, quantile)))

    return n1


def admix_quantiles(n: dict, ev: events.Admix, params: Params, quantile: float = 0.95):
    n1 = n.copy()
    q = ev.f_p(params._demo_dict)

    parent1, parent2 = ev.parent1, ev.parent2
    child = ev.child

    n0 = n[child]

    np1 = int(scipy.stats.binom(n0, q).ppf(quantile))
    np2 = int(scipy.stats.binom(n0, 1 - q).ppf(quantile))

    n1[parent1] = max(np1, 1)
    n1[parent2] = max(np2, 1)
    n1[child] = 0

    return n1


def pulse_quantiles(n: dict, ev: events.Pulse, params: Params, quantile: float = 0.95):
    n1 = n.copy()
    q = ev.f_p(params._demo_dict)

    dest, source = ev.dest, ev.source

    n0 = n[dest]
    n1d = int(scipy.stats.binom(n0, 1 - q).ppf(quantile))
    n1[dest] = max(n1d, 1)
    n1[source] += int(scipy.stats.binom(n0, q).ppf(quantile))

    return n1


def sample_theta_train(loc, scale, size, params):
    A, b, G, h = params._polyhedron_hyperparams()
    X = np.random.normal(loc, scale, size=(size, len(loc))).T
    np.all(np.isclose(A @ X, b[:, None]), 0)
    GT = np.all(G @ X <= h[:, None], 0)
    EQ = np.all(np.isclose(A @ X, b[:, None]), 0)
    keep = GT & EQ
    return X[:, keep]


def bound_sampler(
    T: ETBuilder,
    params: Params,
    size: int,
    loc: jnp.ndarray,
    scale: jnp.ndarray,
    seed: int = None,
    quantile: float = 0.95,
):
    """Bound sampler for event tree

    Args:
        T (ETBuilder): It should be executed
        params (Params): Fitted parameters
        size (int): Sample size for normal random samples
        loc (jnp.ndarray): N(loc, scale) for each params._train_keys
        scale (jnp.ndarray): N(loc, scale) for each params._train_keys
        seed (int, optional): random seed
        quantile (float, optional): Quantile cuts for bounds
    """
    theta_train_sample = np.zeros((len(loc), 0))
    n = T.num_samples
    while theta_train_sample.shape[1] < size:
        theta_train_sample = np.hstack(
            (theta_train_sample, sample_theta_train(loc, scale, size, params))
        )
    theta_train_sample = theta_train_sample[:, :size].T

    nodes = T.nodes
    T = T._T

    NS = {}
    # traverse tree starting at leaves and working up
    for u in nx.topological_sort(T):
        for i, ch in enumerate(T.predecessors(u), 1):
            e = T.edges[ch, u]
            # execute the edge event (if any)
            ev = e.get("event", events.NoOp())

            if (
                isinstance(ev, events.Lift) and not ev.terminal
            ):  # do not bound terminal events
                if ev.migrations:
                    pass
                    # n = Migration_sample(
                    #     n, ev, theta_train_sample, params, seed, quantile
                    # )
                else:
                    n = sample_lift(n, ev, theta_train_sample, params, seed, quantile)
                    NS[ev] = deepcopy(n)
                    print(ev.t1, "Lift", n)
                seed = np.random.RandomState(seed).randint(2**31 - 1)

            elif isinstance(ev, events.Admix):
                n = admix_quantiles(n, ev, params, quantile)
                NS[ev] = deepcopy(n)
                print(u.t, "Admix", n)
            elif isinstance(ev, events.Pulse):
                n = pulse_quantiles(n, ev, params, quantile)
                NS[ev] = deepcopy(n)
                print(u.t, "Pulse", n)
            elif isinstance(ev, events.Rename):
                new, old = ev.new, ev.old
                n[new] = n[old]
                del n[old]
            else:
                pass

        ev = nodes[u].get("event", events.NoOp())
        if isinstance(ev, (events.Split2, events.Split1)):
            donor, recipient = ev.donor, ev.recipient
            if recipient not in n:
                n[recipient] = 0

            n[recipient] += n[donor]
            del n[donor]

        else:
            pass

    return NS
