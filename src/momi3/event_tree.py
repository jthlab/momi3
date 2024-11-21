import math
import operator
from copy import deepcopy
from enum import Enum
from functools import reduce, total_ordering
from itertools import count
from types import ModuleType
from typing import Callable, Iterable, NamedTuple

import demes
import jax.numpy as jnp
import networkx as nx
from frozendict import frozendict
from loguru import logger

from momi3.common import Population, Time, unique_strs


@total_ordering
class EventType(Enum):
    EPOCH = 1
    MIGRATION_END = 2
    MIGRATION_START = 3
    PULSE = 4
    MERGE = 5
    POPULATION_START = 6

    def __lt__(self, other: "EventType") -> bool:
        assert isinstance(other, EventType)
        return self.value < other.value


def _all_events(demo: demes.Graph) -> Iterable[dict]:
    """Iterate over all events in the demes graph"""
    d = demo.asdict()
    for i, deme in enumerate(d["demes"]):
        name = deme["name"]
        for j, e in enumerate(deme["epochs"]):
            # size change events
            path = ("demes", i, "epochs", j, "end_time")
            t = e["end_time"]
            yield dict(
                t=t,
                path=path,
                pop=name,
                size_function=e["size_function"],
                ev=EventType.EPOCH,
                i=j,
            )
        if deme["ancestors"]:
            # merge events
            path = ("demes", i, "start_time")
            yield dict(
                t=deme["start_time"],
                path=path,
                pop=name,
                ancestors=deme["ancestors"],
                i=i,
                ev=EventType.MERGE,
            )
        else:
            # deme has no ancestors, so it must extend infinitely back into the past
            assert math.isinf(deme["start_time"])
            path = ("demes", i, "start_time")
            yield dict(
                t=math.inf,
                path=path,
                pop=name,
                ev=EventType.POPULATION_START,
            )
    # pulse admixtures
    for j, p in enumerate(d["pulses"]):
        path = ("pulses", j, "time")
        yield dict(
            t=p["time"],
            path=path,
            i=j,
            pop=p["dest"],
            sources=p["sources"],
            ev=EventType.PULSE,
        )
    # migration start and stop
    for j, m in enumerate(d["migrations"]):
        y = dict(i=j, source=m["source"], pop=m["dest"])
        # start and end are backwards for us since we are working in reverse time
        path = ("migrations", j, "end_time")
        yield y | dict(t=m["end_time"], path=path, ev=EventType.MIGRATION_START)
        path = ("migrations", j, "start_time")
        yield y | dict(t=m["start_time"], path=path, ev=EventType.MIGRATION_END)


class Node(NamedTuple):
    i: int
    block: frozenset[Population]
    t: Time


class EventTree:
    """Build an event tree from a demes graph.

    Args:
        demo: a demes graph
        num_samples: a dictionary mapping deme names to the number of samples in that deme.
    """

    def __init__(
        self,
        demo: demes.Graph,
        num_samples: dict[str, int],
        events: ModuleType,
    ):
        self._demo = demo
        self._num_samples = num_samples
        self._events = events
        self._T = nx.DiGraph()
        # initialize the event tree
        self._times = {}
        self._init_leaves()
        # compute aux information for each node
        self._build_tree()

    @property
    def events(self):
        return self._events

    def _add_time(self, t, path):
        tm = Time(t, path=path)
        self._times.setdefault(t, set()).add(tm)
        return tm

    def _init_leaves(self):
        # initialize the event tree
        self._i = count(1)
        leaves = self._leaves = {}
        # Initialize leaf nodes for each population
        for j, deme in enumerate(self._demo.demes):
            # add initial leaf nodes for each population
            path = ("demes", j, "epochs", -1, "end_time")
            t = self._add_time(deme.epochs[-1].end_time, path=path)
            node = Node(i=next(self._i), block=frozenset([deme.name]), t=t)
            # attached to each node are attributes that track the population size and migration rates. (these are the
            # two model attributes that persist over time).
            self.add_node(
                node,
                epochs=frozendict({deme.name: len(deme.epochs) - 1}),
                migrations=frozendict(),
            )
            leaves[deme.name] = node

    def setup(self):
        # precompute auxiliary information for each event
        leaves = self.leaves
        events = self.events
        auxd = {"nodes": {}, "edges": {}}

        for u in nx.topological_sort(self._T):
            child_axes = {}
            child_ns = {}
            for i, ch in enumerate(self._T.predecessors(u), 1):
                e = self._T.edges[ch, u]
                ax = self.nodes[ch]["axes"]
                ns = self.nodes[ch]["ns"]
                ev = e.get("event", events.NoOp())
                new_ax, new_ns, aux = ev.setup(ax, ns)
                assert (
                    new_ax.keys() == new_ns.keys()
                )  # the axes and samples should contain the same pops
                assert not (
                    child_ns.keys() & new_ns.keys()
                )  # the new populations should be completely disjoint from earlier ones
                child_ns.update(new_ns)
                auxd["edges"][(ch, u)] = aux
                id_ = e.get("id", f"child{i}") + "_axes"
                child_axes[id_] = new_ax
                e["axes"] = new_ax
            ev = self.nodes[u].get("event", events.NoOp())
            if not child_axes:
                # if there are no children this should be a leaf node
                assert u in leaves.values()
                continue
            if len(child_axes) == 1:
                # if there is only one child axis, the event expects a single positional argument
                child_axes = list(child_axes.values())[0]
            else:
                # the event expects a dictionary indicating which axes belong to which pop
                assert isinstance(ev, (events.MigrationStart, events.Split2))
            (
                new_ax,
                new_ns,
                aux,
            ) = ev.setup(child_axes, child_ns)
            auxd["nodes"][u] = aux
            self.nodes[u].update({"axes": new_ax, "ns": new_ns})
        return auxd

    def execute(self, params: dict, auxd: dict) -> jnp.ndarray:
        """Execute the event tree.

        Args:
            X: a dictionary mapping populations to leaf node values.
            params: a dictionary of model parameters.
            auxd: a dictionary of auxiliary data.

        Returns:
            Expected branch length subtending leaf configurations.
        """
        # assert set(X) == set(self._leaves)
        # initialize leaf node partials
        # traverse tree starting at leaves and working up
        events = self.events
        for u in nx.topological_sort(self._T):
            logger.trace("executing node {}", u)
            child_state = {}
            for i, ch in enumerate(self._T.predecessors(u), 1):
                st = self.nodes[ch]["state"]
                e = self._T.edges[ch, u]
                aux = auxd["edges"].get((ch, u))
                # execute the edge event (if any)
                ev = e.get("event", events.NoOp())
                new_st = ev.execute(st, params=params, aux=aux)
                assert isinstance(new_st, st.__class__)
                id_ = e.get("id", f"child{i}") + "_state"
                child_state[id_] = new_st
                # check that the returned state is consistent with the child axes
                # FIXME
                if new_st.terminal:
                    # the final lifting to infinity makes the partial likelihood None
                    assert isinstance(ev, events.Lift)
                    assert ev.terminal
                else:
                    new_st.check_shape(e["axes"])
            if not child_state:
                # if no child state, it has to be a leaf node
                assert u in self._leaves.values()
                continue
            aux = auxd["nodes"].get(u)
            # NoOp.execute accepts only a single state parameter, and returns it. However, there is no possibility of
            # passing more than one state parameter in, because the only type of event that has multiple children is a
            # Split2.
            ev = self.nodes[u].get("event", events.NoOp())
            assert len(child_state) in [1, 2]
            if len(child_state) == 1:
                child_state = list(child_state.values())[0]
            elif len(child_state) == 2:
                assert isinstance(ev, (events.MigrationStart, events.Split2))
            new_st = ev.execute(child_state, params=params, aux=aux)
            assert isinstance(new_st, st.__class__)
            # FIXME
            if new_st.terminal:
                assert (
                    self._T.out_degree[u] == 0
                )  # this is the root, i.e. the last event to process
            else:
                new_st.check_shape(self.nodes[u]["axes"])
            self.nodes[u]["state"] = new_st
        return self.nodes[u]["state"]

    @property
    def nodes(self):
        return self._T.nodes

    @property
    def edges(self):
        return self._T.edges

    def add_node(self, u: Node, **kw):
        assert isinstance(u, Node)
        self._T.add_node(u, **kw)

    def add_edge(self, u: Node, v: Node, **kw):
        assert isinstance(u, Node)
        assert isinstance(v, Node)
        self._T.add_edge(u, v, **kw)

    def node_like(self, u, i=None, block=None, t=None, **kw) -> Node:
        """return a node which has the same blocks, (optionally) time, and attributes as u"""
        if i is None:
            i = next(self._i)
        ret = Node(i=i, block=block or u.block, t=t or u.t)
        attr = deepcopy(self.nodes[u])
        attr.update(kw)
        try:
            del attr["event"]  # do not copy the event, for it should be different
        except KeyError:
            pass
        self._T.add_node(ret, **attr)
        return ret

    def _get_active(self, pop):
        """get the active (most recent) node for a population"""
        assert nx.is_forest(self._T)
        for u in reversed(list(nx.topological_sort(self._T))):
            if pop in u.block:
                return u

    def _lift(self, pop: Population, t: Time) -> Node:
        """lift node u to time t.

        Args:
            pop: the population to lift
            t: the time to lift to

        Returns:
            the lifted node

        Notes:
            Does nothing if the population is already at time t.
        """
        u = self._get_active(pop)
        if u.t == t:
            # no lifting is necessary
            return u
        assert u.t < t
        # create a new node that is the same as u, but with a different time
        v = self.node_like(u, t=t)
        ev = self.events.Lift(
            t0=u.t,
            t1=v.t,
            epochs=self.nodes[u]["epochs"],
            migrations=self.nodes[u]["migrations"],
        )
        self._T.add_edge(u, v, event=ev)
        return v

    # def bound(self, bounds):
    #     for d in self.nodes, self.edges:
    #         for u in d:
    #             ev = d[u].get("event")
    #             if ev is not None and ev in bounds:
    #                 d[u]["event"] = dataclasses.replace(ev, bounds=bounds[ev])
    #     self._setup()
    #     return self

    def _merge_nodes(self, x: Node, y: Node, rm=None) -> Node:
        """merge nodes x and y, optionally removing rm from the merged block set."""
        assert x.t == y.t
        t = x.t
        # OR together the migration sets and epochs dict
        st = {
            k: reduce(operator.or_, [self.nodes[z][k] for z in (x, y)])
            for k in ["migrations", "epochs"]
        }
        b = x.block | y.block  # new blocks, obtained by merging previous blocks
        if rm:
            b -= {rm}
            if rm in st["epochs"]:
                st["epochs"] = st["epochs"].delete(rm)
        nn = Node(i=next(self._i), block=b, t=t)
        self.add_node(nn, **st)
        for z in x, y:
            self._T.add_edge(z, nn)
        return nn

    def _build_tree(self):
        """build the event tree"""
        events = self.events

        # this sorting function ensures that:
        # - events are processed (reverse-)chronologically
        # - contemporaneous events are processed in the order specified by EventType
        # - contemporaneous events of the same type are processed according to their order specified by demes.
        # the last point matters for simultaneous pulses in particular:
        # https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses
        def keyfun(d):
            # TODO explain why d.get('i') is necessary
            return (d["t"], d["ev"], d.get("i"))

        # iterate over all events in the sort order specified above
        for d in sorted(_all_events(self._demo), key=keyfun):
            # register times of all events, including epochs
            t = self._add_time(d["t"], d["path"])

            # if epoch, nothing to do. epochs are handled by the lifting events.
            if d["ev"] == EventType.EPOCH:
                continue
                # nn = self.node_like(u)
                # self.nodes[nn]["epochs"] = self.nodes[nn]["epochs"].set(
                #     d["pop"], d["i"]
                # )
                # self.add_edge(u, nn)

            # otherwise, lift the population to current time and process event
            u = self._lift(d["pop"], t)
            assert u.t == t

            # migrations update state, and also merge nodes in the event tree
            if d["ev"] == EventType.MIGRATION_START:
                v = self._lift(d["source"], t)
                # sanity checks
                key = (d["source"], d["pop"])
                if v is u:
                    self.nodes[u]["migrations"] = self.nodes[u]["migrations"].set(
                        key, d["i"]
                    )
                else:
                    st_u, st_v = [self.nodes[x] for x in (u, v)]
                    # the nodes should be fully disjoint, otherwise they would already be in the same block
                    assert not (st_u["epochs"].keys() & st_v["epochs"].keys())
                    assert not (st_u["migrations"].keys() & st_v["migrations"].keys())
                    nn = self._merge_nodes(u, v)  # now nn has children u and v
                    # per the demes spec, continuous migrations cannot overlap
                    assert key not in self.nodes[nn]["migrations"]
                    self.nodes[nn]["migrations"] = self.nodes[nn]["migrations"].set(
                        key, d["i"]
                    )
                    self.nodes[nn]["event"] = events.MigrationStart(
                        source=d["source"], dest=d["pop"]
                    )
                    self.edges[u, nn]["id"] = "dest"
                    self.edges[v, nn]["id"] = "source"

            # a state update. the nodes are already in the same block, and remain so even after migration ends.
            elif d["ev"] == EventType.MIGRATION_END:
                key = (d["source"], d["pop"])
                nn = self.node_like(u)
                self.nodes[nn]["migrations"] = self.nodes[nn]["migrations"].delete(key)
                self.add_edge(u, nn)

            # pulses function in a similarly to continuous migrations, but they are not recorded in the state since they
            # happen instantly.
            elif d["ev"] == EventType.PULSE:
                # From https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses  # noqa: E501
                # 1. Initialize an array of zeros with length equal to the number of demes.
                # 2. Set the ancestry proportion of the destination deme to 1.
                # 3. For each pulse:
                #    a. Multiply the array by one (1) minus the sum of proportions.
                #    b. For each source, add its proportion to the array.
                for j, s in enumerate(d["sources"]):

                    def f_p(params, i=d["i"], j=j):
                        return params["pulses"][i]["proportions"][j]

                    self._pulse(source=s, dest=d["pop"], t=t, f_p=f_p)

            elif d["ev"] == EventType.MERGE:
                # the population merges with ancestral population(s). we model this as a sequence of pulses,
                # followed by admixture.
                for j, s in enumerate(d["ancestors"][:-1]):

                    def f_p(params, i=d["i"], j=j):
                        deme = params["demes"][i]
                        p = sum(deme["proportions"][:j])
                        # at the j-th pulse a fraction 1 - p of the population remains to be admixed
                        return deme["proportions"][j] / (1 - p)

                    self._pulse(source=s, dest=d["pop"], t=t, f_p=f_p)
                # the remaining ancestor merges with last ancestor
                s = d["ancestors"][-1]
                u = self._lift(d["pop"], t)
                v = self._lift(s, t)
                nn = self._merge_nodes(u, v, rm=d["pop"])
                evc = events.Split1 if u is v else events.Split2
                self.nodes[nn]["event"] = evc(donor=d["pop"], recipient=s)
                # identify which edge is which for later traversal
                self.edges[u, nn]["id"] = "donor"
                self.edges[v, nn]["id"] = "recipient"

            elif d["ev"] == EventType.POPULATION_START:
                # the population extends infinitely far back into the past. basically just a lifting event.
                pass
                # self._lift(d["pop"], t)

            else:
                raise RuntimeError(f"unknown event type {d['ev']}")

        assert nx.is_tree(self._T)  # sanity check.

    def _pulse(self, source: Population, dest: Population, t: Time, f_p: Callable):
        """forward-in-time pulse from source into dest"""
        events = self.events
        u = self._lift(dest, t)
        v = self._lift(source, t)
        # there are two cases to consider depending on whether they are in the same block or not
        if u is v:
            # same block, so we perform the pulse in one tensor contraction
            w = self.node_like(u)
            self.add_edge(u, w, event=events.Pulse(source=source, dest=dest, f_p=f_p))
        else:
            # different blocks, so we model the pulse as an admixture followed by a split2
            tr1, tr2 = unique_strs(u.block, 2)
            b = (u.block - {dest}) | {
                tr1,
                tr2,
            }  # augment the blocks of u with the new transient pop
            w = self.node_like(u, block=b, t=t)
            self.add_edge(
                u, w, event=events.Admix(child=dest, parent1=tr1, parent2=tr2, f_p=f_p)
            )
            # now we need to merge the transient admixed population into the source population
            x = self._merge_nodes(w, v, rm=tr1)
            self.nodes[x]["event"] = events.Split2(donor=tr1, recipient=source)
            # identify which edge is which for traversal
            self.edges[w, x]["id"] = "donor"
            self.edges[v, x]["id"] = "recipient"
            # finally, rename the transient population to the destination population
            assert x.block == (u.block | v.block | {tr2}) - {dest}
            y = self.node_like(x, block=u.block | v.block)
            self.add_edge(x, y, event=events.Rename(old=tr2, new=dest))

    @property
    def leaves(self):
        return self._leaves

    @property
    def times(self):
        return self._times
