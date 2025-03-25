import math
import operator
from copy import deepcopy
from enum import Enum
from functools import reduce, total_ordering
from itertools import count
from types import ModuleType
from typing import Callable, Iterable, NamedTuple
from collections.abc import Collection

import demes
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logit
import networkx as nx
from frozendict import frozendict
from loguru import logger

from momi3.common import (
    Population,
    Time,
    Path,
    unique_strs,
    get_path,
    inv_softplus,
    inv_softmax,
)


@total_ordering
class EventType(Enum):
    MIGRATION_START = 1
    MIGRATION_END = 2
    EPOCH = 3
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
        events: a module containing event classes
    """

    def __init__(
        self,
        demo: demes.Graph,
        events: ModuleType,
    ):
        self._demo = demo
        self._events = events
        self._paths = set()
        self._T = nx.DiGraph()
        # initialize leaves and then build the event tree
        with jax.disable_jit(True):
            self._init_leaves()
            self._build_tree()

    @property
    def events(self):
        return self._events

    @property
    def leaves(self):
        return self._leaves

    def reparameterize(self, paths: Collection[Path]):
        return _reparameterize_event_tree(self, paths)

    def _init_leaves(self):
        # initialize the event tree
        self._i = count(1)
        leaves = self._leaves = {}
        # Initialize leaf nodes for each population
        for j, deme in enumerate(self._demo.demes):
            # add initial leaf nodes for each population
            n = len(deme.epochs)
            path = ("demes", j, "epochs", n - 1, "end_time")
            t = Time(deme.epochs[n - 1].end_time, path=path)
            self._paths.add(frozenset([path]))
            node = Node(i=next(self._i), block=frozenset([deme.name]), t=t)
            # attached to each node are attributes that track the population size and
            # migration rates. (these are the two model attributes that persist
            # over time).
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
                # if there is only one child axis, the event expects a single argument
                child_axes = list(child_axes.values())[0]
            else:
                # the event expects a dict indicating which axes belong to which pop
                assert isinstance(ev, (events.MigrationStart, events.Split2))
            (
                new_ax,
                new_ns,
                aux,
            ) = ev.setup(child_axes, child_ns)
            auxd["nodes"][u] = aux
            self.nodes[u].update({"axes": new_ax, "ns": new_ns})
        return auxd

    def constraints(self, paths: Collection[Path] = None):
        if paths is None:
            paths = self._paths
        return self.reparameterize(paths).constraints

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
            # NoOp.execute accepts only a single state parameter, and returns it.
            # However, there is no possibility of passing more than one state parameter
            # in, because the only type of event that has multiple children is a Split2.
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
        """return a node which has the same blocks, (optionally) time, and attributes
        as u"""
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

    def _merge_paths(self, p0: Path, p1: Path):
        "merge the blocks containing p0 and p1"
        bl0, bl1 = [next(s for s in self._paths if p in s) for p in (p0, p1)]
        if bl0 is bl1:
            return
        self._paths.remove(bl0)
        self._paths.remove(bl1)
        self._paths.add(bl0 | bl1)

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
        if u.t.t == t.t:
            self._merge_paths(u.t.path, t.path)
            return u
        assert u.t.t < t.t
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
        assert x.t.t == y.t.t
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
            for m in st["migrations"]:
                if rm in m:
                    st["migrations"] = st["migrations"].delete(m)
        nn = Node(i=next(self._i), block=b, t=x.t)
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
        # - contemporaneous events of the same type are processed according to their
        #   order specified by demes.
        # the last point matters for simultaneous pulses in particular:
        # https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses  # noqa: E501
        def keyfun(d):
            # TODO explain why d.get('i') is necessary
            return (d["t"], d["ev"], d.get("i"))

        # iterate over all events in the sort order specified above
        for d in sorted(_all_events(self._demo), key=keyfun):
            # register times of all events, including epochs
            t = Time(d["t"], d["path"])
            self._paths.add(frozenset([t.path]))

            u = self._lift(d["pop"], t)
            assert u.t.t == t.t
            # if epoch, nothing to do. epochs are handled by the lifting events.

            if d["ev"] in (EventType.EPOCH, EventType.MIGRATION_END):
                continue

            elif d["ev"] == EventType.MIGRATION_START:
                key = (d["source"], d["pop"])
                v = self._lift(d["source"], t)
                if u is v:
                    # these populations are all in the same block
                    self.nodes[u]["migrations"] = self.nodes[u]["migrations"].set(
                        key, d["i"]
                    )
                    continue
                st_u, st_v = [self.nodes[x] for x in (u, v)]
                # the nodes should be fully disjoint, otherwise they would already be in
                # the same block
                assert not (st_u["epochs"].keys() & st_v["epochs"].keys())
                assert not (st_u["migrations"].keys() & st_v["migrations"].keys())
                nn = self._merge_nodes(u, v)  # now nn has children u and v
                # per the demes spec, continuous migrations cannot overlap
                # assert (d["source"], d["pop not in self.nodes[nn]["migrations"]
                key = (d["source"], d["pop"])
                self.nodes[nn]["migrations"] = self.nodes[nn]["migrations"].set(
                    key, d["i"]
                )
                self.nodes[nn]["event"] = events.MigrationStart(
                    source=d["source"], dest=d["pop"]
                )
                self.edges[u, nn]["id"] = "dest"
                self.edges[v, nn]["id"] = "source"
                continue

            # a state update. the nodes are already in the same block, and remain so
            # even after migration ends.
            # elif d["ev"] == EventType.MIGRATION_END:
            #     key = (d["source"], d["pop"])
            #     nn = self.node_like(u)
            #     self.nodes[nn]["migrations"] = self.nodes[nn]["migrations"].delete(key)
            #     self.add_edge(u, nn)

            # pulses function in a similarly to continuous migrations, but they are not
            # recorded in the state since they happen instantly.

            elif d["ev"] == EventType.PULSE:
                # From https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses  # noqa: E501
                # 1. Initialize an array of zeros with length equal to the num. demes.
                # 2. Set the ancestry proportion of the destination deme to 1.
                # 3. For each pulse:
                #    a. Multiply the array by one (1) minus the sum of proportions.
                #    b. For each source, add its proportion to the array.
                for j, s in enumerate(d["sources"]):

                    def f_p(params, i=d["i"], j=j):
                        return params["pulses"][i]["proportions"][j]

                    self._pulse(source=s, dest=d["pop"], t=t, f_p=f_p)

            elif d["ev"] == EventType.MERGE:
                # the population merges with ancestral population(s). we model this as a
                # sequence of pulses, followed by admixture.
                for j, s in enumerate(d["ancestors"][:-1]):

                    def f_p(params, i=d["i"], j=j):
                        deme = params["demes"][i]
                        p = sum(deme["proportions"][:j])
                        # at the j-th pulse a fraction 1 - p of the population remains
                        # to be admixed
                        return deme["proportions"][j] / (1 - p)

                    self._pulse(source=s, dest=d["pop"], t=t, f_p=f_p)
                # the remaining ancestor merges with last ancestor
                s = d["ancestors"][-1]
                v = self._lift(s, t)
                if d["pop"] in v.block:
                    # the populations are already in the same block
                    w = self.node_like(v)
                    self.add_edge(
                        v, w, event=events.Split1(donor=d["pop"], recipient=s)
                    )
                else:
                    w = self._merge_nodes(u, v, rm=d["pop"])
                    self.nodes[w]["event"] = events.Split2(donor=d["pop"], recipient=s)
                    # identify which edge is which for later traversal
                    self.edges[u, w]["id"] = "donor"
                    self.edges[v, w]["id"] = "recipient"

            elif d["ev"] == EventType.POPULATION_START:
                # the population extends infinitely far back into the past. basically
                # just a lifting event.
                pass
                # self._lift(d["pop"], t)

            else:
                raise RuntimeError(f"unknown event type {d['ev']}")

        assert nx.is_tree(self._T)  # sanity check.
        self._collapse_successive_lifts()

    def _collapse_successive_lifts(self):
        """collapse successive lift events into a single event"""
        self._full_T = self._T
        self._T = self._full_T.copy()

        def f():
            for u, v in self._T.edges():
                if self._T.in_degree(v) != 1:
                    continue
                succ = list(self._T.successors(v))
                if len(succ) == 0:
                    # root node
                    assert np.isinf(v.t.t)
                    continue
                else:
                    assert len(succ) == 1
                    w = succ[0]

                def edge_is_lift(e):
                    return isinstance(e.get("event"), self.events.Lift)

                if edge_is_lift(self._T.edges[u, v]) and edge_is_lift(
                    self._T.edges[v, w]
                ):
                    # collapse the two lifts into a single lift
                    t0 = self.edges[u, v]["event"].t0
                    t1 = self.edges[v, w]["event"].t1
                    ev = self.events.Lift(
                        t0=t0,
                        t1=t1,
                        epochs=self.nodes[u]["epochs"],
                        migrations=self.nodes[u]["migrations"]
                        | self.nodes[v]["migrations"],
                    )
                    self._T.add_edge(u, w, event=ev)
                    self._T.remove_node(v)
                    # repeat the process until there are no more successive lifts
                    return False
            return True

        # repeat the process until there are no more successive lifts
        while not f():
            pass

        assert nx.is_tree(self._T)  # sanity check.

    def _pulse(self, source: Population, dest: Population, t: Time, f_p: Callable):
        """forward-in-time pulse from source into dest"""
        events = self.events
        u = self._lift(dest, t)
        v = self._lift(source, t)
        # there are two cases to consider depending on whether they are in the same
        # block or not
        if u is v:
            # same block, so we perform the pulse in one tensor contraction
            w = self.node_like(u)
            self.add_edge(u, w, event=events.Pulse(source=source, dest=dest, f_p=f_p))
        else:
            # different blocks, so we model the pulse as an admixture followed by a
            # split2
            tr1, tr2 = unique_strs(u.block, 2)
            b = (u.block - {dest}) | {
                tr1,
                tr2,
            }  # augment the blocks of u with the new transient pop
            w = self.node_like(u, block=b, t=t)
            self.add_edge(
                u, w, event=events.Admix(child=dest, parent1=tr1, parent2=tr2, f_p=f_p)
            )
            # now we need to merge the transient admixed population into the source
            # population
            x = self._merge_nodes(w, v, rm=tr1)
            self.nodes[x]["event"] = events.Split2(donor=tr1, recipient=source)
            # identify which edge is which for traversal
            self.edges[w, x]["id"] = "donor"
            self.edges[v, x]["id"] = "recipient"
            # finally, rename the transient population to the destination population
            assert x.block == (u.block | v.block | {tr2}) - {dest}
            y = self.node_like(x, block=u.block | v.block)
            self.add_edge(x, y, event=events.Rename(old=tr2, new=dest))


def _reparameterize_event_tree(tree: EventTree, paths: Collection[Path]):
    T = tree._full_T
    assert nx.is_directed_acyclic_graph(T)
    assert nx.number_connected_components(T.to_undirected()) == 1

    params0 = tree._demo.asdict()
    paths = set(paths)
    fd = {}
    finvd = {}

    def is_time_path(path):
        return path[-1] in ("start_time", "end_time", "time")

    def get_path_block(path):
        return next(s for s in tree._paths if path in s)

    # validate the list of paths
    for p in paths:
        try:
            get_path(params0, p)
        except KeyError as e:
            raise ValueError(f"path {p} not found in demes graph") from e
    # check that the start time of the first deme is not in the list of paths
    if ("demes", 0, "start_time") in paths:
        assert math.isinf(demes[0]["start_time"])
        raise ValueError(
            "cannot reparameterize the start time of the first deme, "
            "as it extends infinitely far back in the past"
        )

    def f_pos(x, _):
        return jax.nn.softplus(x)

    def finv_pos(y, _):
        return inv_softplus(y)

    def f_simplex(x, _):
        return jax.nn.softmax(x)

    def finv_simplex(y, _):
        return inv_softmax(y)

    def f_01(x, _):
        return jax.nn.sigmoid(x)

    def finv_01(y, _):
        return jax.scipy.special.logit(y)

    # list of constraints
    constraints = []

    # check no duplication in the path list
    for path in paths:
        # only time paths can be multiply referenced
        if is_time_path(path):
            try:
                other_paths = get_path_block(path) - {path}
            except StopIteration:
                raise ValueError(
                    f"path {path} not found in shared paths, this is a bug!"
                )
            if other_paths & paths:
                raise ValueError(
                    f"cannot reparameterize {path} and {other_paths} "
                    "simultaneously, since they are constrained to be equal"
                )

    # first match all non-time paths
    for path in filter(lambda x: not is_time_path(x), paths):
        if path[-2] == "proportions":
            raise NotImplementedError(
                "I can't reparameterize individual proportions. Instead of passing "
                f"{path}, pass {path[:-1]} to reparameterize the entire vector of "
                "proportions."
            )
        fp = frozenset([path])
        match path[-1]:
            case "proportions":
                fd[fp] = f_simplex
                finvd[fp] = finv_simplex
                constraints.append((path, "simplex"))
            case "rate":
                fd[fp] = f_pos
                finvd[fp] = finv_pos
                constraints.append((path, "[0,1]"))
            case "start_size" | "end_size":
                func_type = get_path(params0, path[:-1] + ("size_function",))
                if func_type == "constant":
                    # if the size function is constant, the start size and end size are
                    # constrained to be equal
                    fp = frozenset(
                        [path[:-1] + (f"{x}_size",) for x in ("start", "end")]
                    )
                fd[fp] = f_pos
                finvd[fp] = finv_pos
                constraints.append((path, "positive"))
            case _:
                raise ValueError(f"unrecognized path {path}")

    # now handle the time paths which are weirder

    # start at root
    Tr = T.reverse()  # edges pointing away from root
    nodes = nx.topological_sort(Tr)
    root = next(nodes)
    assert root.t.t == math.inf
    n = next(nodes)  # this is the "crown" of the tree
    # the crown is special because the time is unbounded above, so it needs to
    # transform to a positive value
    path_block = get_path_block(n.t.path)
    if path_block & paths:
        fd[path_block] = f_pos
        finvd[path_block] = finv_pos
        constraints.append((n.t.path, "positive"))

    # recurse down the tree. each parameterized time node is expressed in
    # terms of a fraction of the time of its nearest ancestor.
    for n in nodes:
        if n.t.path not in paths:
            # this time is not in the list of paths to reparameterize
            continue
        path_block = get_path_block(n.t.path)
        if path_block in fd:
            # this time has already been reparameterized
            continue
        p = n
        while True:
            ps = list(Tr.predecessors(p))
            assert len(ps) == 1
            (p,) = ps
            if p.t.t > n.t.t:
                break

        def f(x, params=params0, parent_path=p.t.path):
            alpha = jax.nn.sigmoid(x)
            return alpha * get_path(params, parent_path)

        fd[path_block] = f

        def finv(y, params=params0, parent_path=p.t.path):
            return logit(y / get_path(params, parent_path))

        finvd[path_block] = finv

        constraints.append((n.t.path, "<=", p.t.path))

    for path_block in fd:
        assert path_block in finvd
        if len(path_block) > 1:
            # all paths in the block should be equal
            # set prints nicer
            constraints.append((set(path_block), "equal"))

    # create return functions that apply the reparameterization and inverse
    # based on the lists created above.
    def f_combined(x, params=params0, fd=fd):
        x = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float64), x)
        ret = {}
        for paths, fp in fd.items():
            # any times which are identically equal in the base model
            # are constrained to be equal during reparameterization
            val = fp(x[paths], params)
            for path in paths:
                ret[path] = val
        return ret

    def finv_combined(params, finvd=finvd):
        ret = {}
        for paths, fi in finvd.items():
            path = next(iter(paths))
            # all paths in the block should be equal
            y = jnp.array(get_path(params, path), dtype=jnp.float64)
            ret[paths] = fi(y, params)
        return ret

    f_combined.constraints = constraints
    return f_combined, finv_combined(params0)
