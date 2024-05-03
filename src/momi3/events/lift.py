"""Lift an event backwards in time"""
import itertools as it
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

import networkx as nx
from jax import numpy as jnp
from jax import vmap

from momi3.common import (
    Axes,
    Ne_t,
    PopCounter,
    Population,
    State,
    Time,
    oe_einsum,
    traverse,
)
from momi3.math_functions import exp_integral, exp_integralEGPS, expm1d
from momi3.migration import lift_cm, lift_cm_aux
from momi3.utils import W_matrix, moran_eigensystem, rate_matrix

from .event import Event

T = TypeVar("T")


def _aux_single(nv):
    d, Q = moran_eigensystem(nv)
    M = rate_matrix(nv).toarray()
    QQ, RR = jnp.linalg.qr(Q)
    W = W_matrix(nv).astype(float)
    return dict(d=d, Q=Q, M=M, W=W, QQ=QQ, RR=RR)


@dataclass(frozen=True, kw_only=True)
class Lift(Event):
    """Lift a partial likelihood from u to v.

    Attributes:
        t0: time at u (closer to present)
        t1: time at v (more ancient)
        epochs: current epoch information at u
        migrations: current migration information at u
    """

    t0: Time
    t1: Time
    epochs: dict[Population, int]
    migrations: dict[tuple[Population, Population], int]

    @property
    def terminal(self):
        return math.isinf(self.t1.t)

    def _setup_impl(
        self, child_axes: Axes, ns: PopCounter
    ) -> tuple[Axes, PopCounter, T]:
        """Compute the matrices needed for lifting.

        Args:
            child_dims: dimensions of the child partial likelihood.
            nv: number of leaf nodes subtended by each population at the lower node

        Returns:
            dims: dimensions of the resulting partial likelihood.
            aux: dict mapping each migration set to the necessary parameters for lifting
        """
        G = nx.DiGraph()
        G.add_nodes_from(
            child_axes
        )  # add a node for each population in the child partial likelihood
        G.add_edges_from(self.migrations)
        migration_sets = [tuple(c) for c in nx.connected_components(G.to_undirected())]
        # refine migration sets to only consider who is migrating into who
        migration_sets1 = []
        for cc in migration_sets:
            if len(cc) == 1:
                # if there is no migration, lift the single population in isolation
                migration_sets1.append(cc[0])
                continue
            a = []
            for c1, c2 in it.product(cc, repeat=2):
                if G.has_edge(c1, c2):
                    a.append((c1, c2))
            migration_sets1.append(tuple(a))
        aux = {"mats": {"single": {}, "multi": {}}, "axes": child_axes}
        nsp = deepcopy(ns)
        for s in migration_sets1:
            if isinstance(s, Population):
                # 1d lifting
                pop = s
                nv = child_axes[pop] - 1
                assert nv == sum(
                    ns[pop].values()
                )  # the dimension of the child axes should match what is tracked by ns
                # without migration, the number of subtended leaf lineages does not change
                # print(pop, nv)
                aux["mats"]["single"][s] = _aux_single(nv)
            else:
                # lifting with multiple populations
                cmm = lift_cm_aux(child_axes, s)
                aux["mats"]["multi"][s] = dict(cmm=cmm)

        # set up functions for computing migration rates and pop sizes at runtime
        return child_axes, nsp, aux

    def _f_Ne(self, params: dict) -> dict[Population, tuple[float, float]]:
        """get the population size at the start and end of the interval"""
        deme_d = {deme["name"]: i for i, deme in enumerate(params["demes"])}
        ret = {}
        for pop in self.epochs:
            i = deme_d[pop]
            j = self.epochs[pop]
            N0, N1 = tuple(
                [
                    _get_size(params["demes"][i], j, traverse(params, path))
                    for path in (self.t0.path, self.t1.path)
                ]
            )
            if params["demes"][i]["epochs"][j]["size_function"] == "constant":
                ret[pop] = N0
            else:
                ret[pop] = (N0, N1)
        return ret

    def _migmat(
        self, params: dict, s: list[tuple[Population, Population]]
    ) -> dict[tuple[Population, Population], float]:
        """get the migration matrix for a block"""
        ret = {}
        for (p1, p2), j in self.migrations.items():
            if (p1, p2) in s:
                m = params["migrations"][j]
                assert m["source"] == p1 and m["dest"] == p2
                ret[(p1, p2)] = params["migrations"][j]["rate"]
        return ret

    def _execute_impl(self, st: State, params: dict, aux: T) -> State:
        """Lift partial likelihood.

        Args:
            st: state just before the lifting event
            params: dict of parameters
            f: function which takes the parameters dict and returns N0, N1, tau
               (N0=starting population size, N1=ending population size, tau=elapsed time)
            aux: dict of auxiliary data generated by setup

        Returns:
            State after the lifting event.
        """
        plp = st.pl
        phip = 0.0
        size_d = self._f_Ne(params)
        t1_val = traverse(params, self.t1.path)
        t0_val = traverse(params, self.t0.path)
        tau = t1_val - t0_val
        axes = aux["axes"]
        for mat_type in ("single", "multi"):
            for s in aux["mats"][mat_type]:
                if mat_type == "single":
                    assert isinstance(s, str)
                    mats = aux["mats"]["single"][s]
                    assert isinstance(s[0], Population)
                    pop = s
                    involved_pops = {pop}
                    i = list(axes).index(pop)
                    Ne = size_d[pop]
                    plp, etbl = _lift1(
                        plp,
                        i,
                        Ne,
                        tau,
                        mats["d"],
                        mats["Q"],
                        mats["M"],
                        mats["QQ"],
                        mats["RR"],
                        mats["W"],
                        self.terminal,
                    )
                else:
                    M = self._migmat(params, s)
                    mats = aux["mats"]["multi"][s]
                    involved_pops = {x for ab in s for x in ab}
                    Ne = {pop: size_d[pop] for pop in involved_pops}
                    plp, etbl = _liftmulti(
                        plp, axes, Ne, (t0_val, t1_val), M, mats["cmm"]
                    )
                inds = [0] * st.pl.ndim
                for pop in involved_pops:
                    inds[list(axes).index(pop)] = slice(None)
                pl0 = st.pl[tuple(inds)].squeeze()
                phip += (pl0 * etbl).sum()
        # print(self.t1, self.t0, axes, phip)
        return st._replace(pl=plp, phi=st.phi + phip)


def _lift1(pl, in_axis, Ne, tau, d, Q, M, QQ, RR, W, terminal):
    """
    Lift a partial likelihood along a single axis.
    Args:
        pl: The partial likelihood to lift
        in_axis: axis along which to lift
        Ne: effective population size (if constant) or (N0, N1) size at the bottom/top of the branch
        tau: elapsed time along branch.
        d, Q, W: precomputed matrices; see setup()
        terminal: if true, do not perform lifting

    Returns:
        pl: lifted partial likelihood
        etbl: branch length subtended by the lifted axis
    """
    # first compute phi, the total branch length. do this first because we need to mulitply by the partial likelihood at
    # the bottom.
    nv = pl.shape[in_axis] - 1
    if terminal:
        # FIXME assume constant pop size in terminal/stem branch.
        # this should be enforced by demes anyways, but maybe check earlier in the code.
        j = jnp.arange(2, nv + 1)
        # expected time to coal with pop size N0
        cm = (2 * Ne) / (j * (j - 1) / 2)
        fn = W @ cm
        etbl = jnp.r_[0, fn, 0]
        return None, etbl
    etbl, R = _etbl_R(nv, Ne, tau, W)
    # now compute the lifted partial likelihood
    # we basically want to contract the partial likelihood along the lifted axis with the matrix
    # Q * exp(d * R) * Qinv. however for numerical & computational reasons, avoid matrix-matrix products or inversion
    ed = jnp.exp(R * d)
    pl_axes = list(range(pl.ndim))
    out_axes = list(pl_axes)
    i = max(pl_axes) + 1
    out_axes[in_axis] = i
    # the next line is equivalent to tensordot(pl, d[:, None] * Q, axes=(in_axis, 1))
    Ql = oe_einsum(pl, pl_axes, ed, [i], Q, [i, in_axis], out_axes)

    # the next two lines push the lifted axis to the end and apply a batched solve
    # equivalent to tensordot(Ql, Qinv, axes=(in_axis, 1)) but without forming Qinv
    # def f(x):
    #     # return lax.linalg.triangular_solve(RR, QQ.T @ x, left_side=True, lower=False)
    #     return jnp.linalg.solve(Q, x)

    plp = jnp.apply_along_axis(lambda x: jnp.linalg.solve(Q, x), in_axis, Ql)
    return plp, etbl


def _liftmulti(pl, axes, Ne, t, mig_mat, cmm):
    """Lift multiple populations who are migrating continuously."""
    params = {"Ne": Ne, "mig": mig_mat}
    return lift_cm(params, t, pl, axes, cmm)


def _etbl_R(nv, Ne, tau, W):
    """Total branch length subtended by the lifted axis.

    Args:
        nv: number of lineages
        Ne: starting/ending population size
        tau: elapsed time

    Returns:
       Expected branch length subtending j=2, ..., nv lineages.
    """
    # N1 = N0 * exp(-g * tau) => g = -log(N1 / N0) / tau
    j = jnp.arange(2, nv + 1)
    jC2 = j * (j - 1) / 2.0
    f_const = vmap(exp_integral, (None, None, 0))
    if isinstance(Ne, tuple):
        N1, N0 = Ne
        g = -(jnp.log(N0) - jnp.log(N1)) / tau
        f_exp = vmap(partial(exp_integralEGPS, g), (None, None, 0))
        # cm = lax.cond(jnp.isclose(g, 0.0), f_const, f_exp, 1 / (2 * N1), tau, jC2)
        # R = lax.cond(jnp.isclose(g, 0.0), _R_const, partial(_R_exp, g), 2 * N1, tau)
        cm = jnp.where(
            jnp.isclose(g, 0.0),
            f_const(1 / (2 * N1), tau, jC2),
            f_exp(1 / (2 * N1), tau, jC2),
        )
        R = jnp.where(
            jnp.isclose(g, 0.0), _R_const(2 * N1, tau), partial(_R_exp, g)(2 * N1, tau)
        )
    else:
        cm = f_const(1 / (2 * Ne), tau, jC2)
        R = _R_const(2 * Ne, tau)
    fn = W @ cm
    k = j - 1
    e_tmrca_min_tau = ((k / nv) * fn).sum()
    etbl = jnp.r_[0, fn, jnp.where(jnp.isinf(tau), 0, tau - e_tmrca_min_tau)]
    return etbl, R


def _R_exp(g, N0, tau):
    tau0 = jnp.isclose(tau, 0.0)
    tau_safe = jnp.where(tau0, 1.0, tau)
    ret = jnp.where(tau0, 0.0, expm1d(tau_safe * g) * tau_safe / N0)
    return ret


def _R_const(N0, tau):
    return tau / N0


def _get_size(deme, j, t):
    "get the size at time t for a epoch j defined by start and end times and sizes"
    # time runs backwards for us
    ep = deme["epochs"][j]
    if ep["size_function"] == "constant":
        return ep["start_size"]
    Ne0 = ep["start_size"]
    if j == 0:
        # this key may not exist if the epoch goes back to infinity, but then the size function would have
        # to be constant, so we would already have returned
        t1 = deme["start_time"]
    else:
        t1 = deme["epochs"][j - 1]["end_time"]
    Ne1 = ep["end_size"]
    t0 = ep["end_time"]
    # start_size = end_size * exp(g * (end_time - start_time)) => g = log(Ne0/Ne1) / (t0-t1)
    return Ne_t(Ne0, Ne1, t0, t1, t)


@dataclass(frozen=True, kw_only=True)
class MigrationStart(Event):
    "Class marking the starting of a migration event in populations with different partial likelihoods."
    source: Population
    dest: Population

    def _setup_impl(
        self, in_axes: dict[str, Axes], ns: PopCounter
    ) -> tuple[Axes, PopCounter, T]:
        assert len(in_axes) == 2
        assert in_axes["source_axes"].keys() & in_axes["dest_axes"].keys() == set()
        ax = Axes()
        ax.update(in_axes["source_axes"])
        ax.update(in_axes["dest_axes"])
        return ax, ns, None

    def _execute_impl(self, state: dict[str, State], params: dict, aux: T) -> State:
        st1 = state["source_state"]
        st2 = state["dest_state"]
        pl1 = st1.pl
        pl2 = st2.pl
        tup1 = (...,) + (None,) * pl2.ndim
        tup2 = (None,) * pl1.ndim + (...,)
        plp = pl1[tup1] * pl2[tup2]
        phip = st2.phi * st1.l0 + st1.phi * st2.l0
        l0p = st1.l0 * st2.l0
        return State(pl=plp, phi=phip, l0=l0p)
