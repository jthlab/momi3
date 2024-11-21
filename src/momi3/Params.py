import json
import math
import pprint
import re
from collections import UserDict
from copy import deepcopy
from itertools import count
from math import inf, isinf
from typing import Callable, NamedTuple

import demes
import demesdraw
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import LinearConstraint, linprog

from momi3.event_tree import EventTree
from momi3.polyhedron import project_polyhedron

NoneType = type(None)
NP_max = np.finfo(np.float32).max
# minimal time difference, in generations, to avoid 0 division
MIN_TIME_DIFFERENCE = 0.01


class Params(UserDict):
    """Parameters in a demographic model.

    Attributes:
        add_linear_constraint (func): Adds a new user constraint
    """

    def __init__(self, demo: demes.Graph, T: EventTree):
        UserDict.__init__(self)
        self._demo = demo
        self._T = T
        demo_dict = demo.asdict()
        self._demo_dict = demo_dict
        # serialize the demo_dict, safer than deepcopy
        self._params_json = json.dumps(demo_dict)
        self._user_constraints = []

        # Create time keys dictionary: tau_0, tau_1, ..., tau_p.
        # time_0 < time_1 < ... < time_p
        # tkeys[time_i] = tau_i
        ts = sorted(list(T.times))
        self._time_keys = tkeys = dict(zip(ts, [f"tau_{i}" for i in range(len(ts))]))

        # iter size, rate and prop parameters.
        # Note: Not iterating time parameters because same time points have the same time key (Generated above).
        iter_size, iter_rate, iter_prop = [count() for _ in range(3)]

        # 1 Demes:
        demes_event = "demes"
        for i, deme in enumerate(self._demo_dict["demes"]):
            j = None
            k = None

            # Start time of a Deme:
            param_name = "start_time"
            x = float(deme[param_name])
            key = tkeys[x]
            self._init_param(
                x, key, demes_event, param_name, i, j, k, fixed=np.isinf(x)
            )

            # Proportion of Ancestors:
            param_name = "proportions"
            # If only a single ancestor, proportion is fixed to 1 so there
            # is no parameter.
            if len(deme["proportions"]) > 1:
                for k, proportion in enumerate(deme["proportions"]):
                    x = float(proportion)
                    key = f"pi_{next(iter_prop)}"
                    self._init_param(x, key, demes_event, param_name, i, j, k)

            # Iterate epochs
            for j, epoch in enumerate(deme["epochs"]):
                # End time of an Epoch:
                param_name = "end_time"
                x = float(epoch[param_name])
                key = tkeys[x]
                # do not add a time parameter for the last epoch of a sampled deme.
                # rationale: where 'end_time' is the non-learnable sampling time
                fixed = all(
                    [
                        deme["name"] in self._T._num_samples,
                        j == len(deme["epochs"]) - 1,
                    ]
                )
                self._init_param(x, key, demes_event, param_name, i, j, k, fixed)
                # Size of an Epoch:
                if epoch["size_function"] == "constant":
                    val = next(iter_size)
                    keys = 2 * [f"eta_{val}"]
                else:
                    # Exponential pop size, default _Theta key
                    keys = [f"eta_{next(iter_size)}", f"eta_{next(iter_size)}"]
                for key, param_name in zip(keys, ["start_size", "end_size"]):
                    x = float(epoch[param_name])
                    self._init_param(x, key, demes_event, param_name, i, j, k)

        # 2 Migrations:
        j = None
        k = None
        migration_rates = {}
        demes_event = "migrations"
        for i, migration in enumerate(self._demo_dict["migrations"]):
            # Time of a Migration:
            for param_name in ["start_time", "end_time"]:
                x = float(migration[param_name])
                key = tkeys[x]
                self._init_param(x, key, demes_event, param_name, i, j, k)

            # Rate of a Migration:
            # If A->B and B->A at the same time frame. We use a symmetric migration rate
            sorted_mig = tuple(
                sorted([migration["source"], migration["dest"]])
                + [migration["start_time"], migration["end_time"], migration["rate"]]
            )
            if sorted_mig in migration_rates:
                key = migration_rates[sorted_mig]
            else:
                key = f"rho_{next(iter_rate)}"
                migration_rates[sorted_mig] = key
            param_name = "rate"
            x = float(migration[param_name])
            self._init_param(x, key, demes_event, param_name, i, j, k)

        # 3 Pulses:
        demes_event = "pulses"
        for i, pulse in enumerate(self._demo_dict["pulses"]):
            j = None
            k = None

            # Time of the pulse
            param_name = "time"
            x = float(pulse[param_name])
            key = tkeys[x]
            self._init_param(x, key, demes_event, param_name, i, j, k)

            # Proportions of the pulse
            param_name = "proportions"
            for k, proportion in enumerate(pulse["proportions"]):
                x = float(proportion)
                key = f"pi_{next(iter_prop)}"
                self._init_param(x, key, demes_event, param_name, i, j, k)
                pulse[param_name][k] = x

    def set_train(
        self,
        times: bool = False,
        proportions: bool = False,
        rates: bool = False,
        sizes: bool = False,
    ):
        """Set training status of parameters.

        Args:
            times: Set training status of time parameters.
            proportions: Set training status of proportion parameters.
            rates: Set training status of rate parameters.
            sizes: Set training status of size parameters.

        Notes:
            If a parameter is not set to train, it will be fixed to the value currently specified.
        """
        for key in self:
            if key.startswith("tau"):
                b = times
            elif key.startswith("pi"):
                b = proportions
            elif key.startswith("rho"):
                b = rates
            elif key.startswith("eta"):
                b = sizes
            if not isinstance(self[key], FixedParam):
                self[key].train = b

    @property
    def trainable(self):
        return [key for key in self if self[key].train]

    @property
    def constraints(self):
        # Linear time constraints for times
        cons = set()
        tkeys = self._time_keys
        for t0, t1 in self._T.edges():
            i = t0.t.t
            j = t1.t.t
            ki = tkeys[i]
            kj = tkeys[j]
            if i != j:
                if not self[ki].train:
                    ki = self[ki].value
                if not self[kj].train:
                    kj = self[kj].value
                    if np.isinf(kj):
                        continue
                cons.add(f"{ki}<={kj}-{MIN_TIME_DIFFERENCE}")

        # upper/lower bound constraints
        for key in self.trainable:
            if np.isfinite(self[key].lower_bound):
                cons.add(f"{self[key].lower_bound}<={key}")
            if np.isfinite(self[key].upper_bound):
                cons.add(f"{key}<={self[key].upper_bound}")

        # proportions sum to one constraints
        for i, deme in enumerate(self._demo_dict["demes"]):
            if len(deme["proportions"]) > 1:
                keys = [f"pi_{k}" for k in range(len(deme["proportions"]))]
                cons.add(f"{'+'.join(keys)}==1")

        return LinearConstraints(tuple(self.trainable), cons)

    def __setitem__(self, key: str, value: float):
        assert key in self
        self[key].value = value

    def from_desc(self, desc):
        for k, v in self.items():
            for d in v.paths.values():
                if desc == d:
                    return v
        raise KeyError(f"Parameter with description {desc} not found")

    def update(self, d: dict[str, float]) -> "Params":
        """Update parameters with new default values.

        Args:
            d: A dictionary mapping parameter keys to default values.

        Returns:
            Modified Params with new default values.
        """
        p_new = Params(self._demo, self._T)
        for key, value in d.items():
            p_new[key] = value
        return p_new

    def to_path_dict(self) -> dict[tuple, float]:
        """Returns a dictionary of mapping paths to value"""
        ret = json.loads(self._params_json)
        for key in self:
            for path in self[key].paths:
                set_path(ret, path, self[key].value)
        return ret

    @property
    def _keys(self):
        return sorted(list(self.keys()))

    @property
    def _Paths(self):
        keys = self._keys
        bools = self._train_bool
        paths_train = [tuple(self[key].paths) for key, b in zip(keys, bools) if b]
        paths_nuisance = [
            tuple(self[key].paths) for key, b in zip(keys, bools) if not b
        ]
        paths = tuple(paths_train + paths_nuisance)
        return paths

    def transform_fns(self, val, ptype, inverse=False):
        if ptype == "eta":
            return val

        elif ptype == "tau":
            # it's actually log differences of tau
            if inverse:
                # ret = softplus_inverse(val)
                ret = math.exp(val)
            else:
                # ret = softplus(val)
                ret = math.log(val)

        elif ptype in ["rho", "pi"]:
            if inverse:
                ret = 1 / (1 + math.exp(-val))
            else:
                ret = math.log(val / (1 - val))

        else:
            raise ValueError(f"Unknown {ptype=}")

        return float(ret)

    def _check_parameter(self, key):
        if key not in self:
            keys = self._keys
            raise KeyError(
                "Parameter Not Found. Parameters: {params}".format(
                    params=", ".join(keys)
                )
            )

    @property
    def demo_dict(self):
        return deepcopy(self._demo_dict)

    @property
    def demo_graph(self):
        return demes.Builder.fromdict(self.demo_dict).resolve()

    def size_history(self, **kwargs):
        dG = self.demo_graph
        demesdraw.size_history(dG, **kwargs)

    def key_to_tex(self, key, val=None):
        letter, no = key.split("_")
        key = f"\\{letter}_{{{no}}}"
        if val is None:
            ret = r"$%s$" % key
        else:
            if letter == "pi":
                val += r"\%"
            ret = r"$%s=%s$" % (key, val)
        return ret

    def _solve_y_conflict(self, text_params, log_time):
        positions = set([])
        for key in text_params:
            cur = text_params[key]
            ymin, ymax = cur["ymin"], cur["ymax"]
            r = ymax - ymin
            i = 2
            cont = True
            while cont:
                for j in range(1, i):
                    if log_time:
                        position = 1 + ymin + r * (10**j) / (10**i)
                    else:
                        position = ymin + r * j / i
                    if positions.issuperset({float(position)}):
                        pass
                    else:
                        cont = False
                        positions.add(float(position))
                        break
                i += 1
            cur["y"] = position
            del cur["ymin"]
            del cur["ymax"]

    def tubes(
        self,
        show_values: bool = True,
        show_letters: bool = False,
        USER_DICT: dict[str, float] = None,
        color_intensity_function: Callable = lambda x: x,
        hide_non_inferreds: bool = False,
        fontsize: float = None,
        tau_font_size: float = None,
        pformats: dict[str, Callable] = None,
        tau_keys: list = None,
        nudge_text_pos: dict = {},
        **kwargs,
    ):
        """Customized demesdraw.tubes function. If USER_DICT is none, parameter boxes will be
        green if user set_train'ed the parameter.

        Args:
            show_values (bool, optional): Show values of our parameters on the plot
            show_letters (bool, optional): Show names of our parameters on the plot
            USER_DICT (dict[str, float], optional): A dict of paramters.
                This will be used for color intersity of the boxes
            color_intensity_function (Callable, optional): This will only be used if user defines USER_DICT
                Redness of the box of the parameter = color_intensity_function(USER_DICT[param_key])
            **kwargs: kwargs for demesdraw.tubes
        """
        show_all = not hide_non_inferreds
        dG = self.demo_graph
        ret = demesdraw.tubes(dG, **kwargs)

        path_to_param = {p: k for k in self for p in self[k].paths}

        # print format for vars
        if pformats is None:
            pformats = {
                "eta": lambda x: f"{int(x):d}",
                "tau": lambda x: f"{int(x):d}",
                "rho": lambda x: f"{x:.2g}",
                "pi": lambda x: f"{(100*x):.2f}",
            }

        lxlim, rxlim = ret.get_xlim()
        min_time, max_time = ret.get_ylim()

        log_time = kwargs.get("log_time", False)

        if USER_DICT is not None:
            box_color_by = "USER_DICT"
        else:
            box_color_by = "train"

        default_kwargs = {
            "kwargs": {"va": "bottom", "ha": "center", "fontsize": fontsize}
        }

        text_params = {}

        # PULSES
        times = [i.time for i in dG.pulses]
        pulse_text = []
        for line in ret.get_lines():
            x = line._x
            y = line._y
            style = line.get_linestyle()

            if (style == "--") & (len(x) == 2):
                for j, t in enumerate(times):
                    if jnp.isclose(t, y[0]):
                        x = (x[0] + x[1]) / 2
                        y = y[0]
                        y = np.where((y < 1) & log_time, 1 + y, y)

                        path = ("pulses", j, "proportions", 0)
                        key = path_to_param[path]
                        text_params[key] = {
                            "type": "pi",
                            "x": x,
                            "y": y,
                            "inferred": key in self.trainable,
                        }
                        text_params[key].update(default_kwargs)

                        text = pformats["pi"](dG.pulses[j].proportions[0])
                        pulse_text.append({"x": x, "y": y, "text": text})
                        break

        # # MIGRATIONS
        # times = [i.time for i in dG.pulses]
        # pulse_text = []
        # for i, line in enumerate(ret.get_lines()):
        #     x = line._x
        #     y = line._y
        #     if len(y) == 2:
        #         for j, t in enumerate(times)
        #             if jnp.isclose(t, y[0]):
        #                 x = (x[0] + x[1]) / 2
        #                 y = y[0]
        #                 y = np.where((y < 1) & log_time, 1 + y, y)

        #                 path = ("pulses", j, "proportions", 0)
        #                 key = self._path_to_params[path]
        #                 text_params[key] = {
        #                     "type": "pi",
        #                     "x": x,
        #                     "y": y,
        #                     "inferred": key in self._train_keys,
        #                 }
        #                 text_params[key].update(default_kwargs)

        #                 text = signif(dG.pulses[j].proportions[0])
        #                 pulse_text.append({"x": x, "y": y, "text": text})
        #                 break

        # DEMES
        demes_x_locs = {}

        D = ret.get_xticklabels() + ret.texts

        for d in D:
            demes_x_locs[d._text] = d._x

        for i, d in enumerate(dG.demes):
            x = demes_x_locs[d.name]
            for j, epoch in enumerate(d.epochs):
                # constant population size
                ys = [epoch.start_time, epoch.end_time]
                texts = [epoch.start_size, epoch.end_size]

                if texts[0] == texts[1]:
                    text = pformats["eta"](texts[1])
                    y = ys[1]
                    y = np.where((y < 1) & log_time, 1 + y, y)

                    path = ("demes", i, "epochs", j, "end_size")
                    key = path_to_param[path]
                    text_params[key] = {
                        "type": "eta",
                        "x": x,
                        "y": y,
                        "inferred": key in self.trainable,
                    }
                    text_params[key].update(default_kwargs)

                else:
                    # exponential growth
                    var_type = iter(["start_size", "end_size"])
                    if not ys[0] > max_time:
                        for k in range(2):
                            text = pformats["eta"](texts[k])
                            y = ys[k]
                            y = np.where((y < 1) & log_time, 1 + y, y)
                            # y = np.clip(y, a_min=step, a_max=max_time)
                            if k == 0:
                                va = "top"
                            else:
                                va = "bottom"

                            path = ("demes", i, "epochs", j, next(var_type))
                            key = path_to_param[path]
                            text_params[key] = {
                                "type": "eta",
                                "x": x,
                                "y": y,
                                "inferred": key in self.trainable,
                            }
                            text_params[key].update(default_kwargs)
                            text_params[key] = deepcopy(text_params[key])
                            text_params[key]["kwargs"].update({"va": va})

        rho_keys = sorted([key for key in self if (key[:3] == "rho")])
        # MIGRATIONS
        mig_params = {}
        for key in rho_keys:
            mig_params[key] = {"inferred": self[key].train}
            val = pformats["rho"](self[key].value)
            mig_path = list(list(self[key].paths)[0])
            start_time = mig_path[:-1] + ["start_time"]
            end_time = mig_path[:-1] + ["end_time"]
            st_key = path_to_param[tuple(start_time)]
            en_key = path_to_param[tuple(end_time)]
            mig_params[key] = {
                "type": "rho",
                "x": rxlim,
                "ymin": self[en_key].value,
                "ymax": self[st_key].value,
                "inferred": key in self.trainable,
            }
            mig_params[key].update(default_kwargs)
            mig_params[key] = deepcopy(mig_params[key])

        self._solve_y_conflict(mig_params, log_time)
        text_params.update(mig_params)

        if tau_keys is None:
            tau_keys = sorted(
                [
                    key
                    for key in self
                    if (key[:3] == "tau") & (not isinf(self[key].value))
                ]
            )
        else:
            pass

        non_tau_keys = sorted([key for key in self if key[:3] != "tau"])

        for key in tau_keys:
            text_params[key] = {"inferred": self[key].train}

        colors = [(1.0, 1.0, 1.0), (1.0, 0, 0)]
        cm = mcolors.LinearSegmentedColormap.from_list("Custom", colors)
        for key in text_params:
            box_color = "darkgray"
            color = "white"

            if box_color_by == "train":
                # Color by inference
                if text_params[key]["inferred"]:
                    box_color = "lightgreen"
                    color = "black"
                else:
                    pass

            elif box_color_by == "USER_DICT":
                if key in USER_DICT:
                    box_color = cm(color_intensity_function(USER_DICT[key]))
                    color = "black"
                else:
                    pass

            else:
                pass

            text_params[key]["box_color"] = box_color
            text_params[key]["color"] = color

        prms_box = dict(
            boxstyle="round", fc="lightgray", ec="black", alpha=0.75, pad=0.1
        )

        # PLOTTING FOR ETA, RHO AND PI
        for key in non_tau_keys:
            cur = text_params[key]
            val = pformats[cur["type"]](self[key].value)
            kwargs = cur["kwargs"]
            prms_box_current = deepcopy(prms_box)

            if show_letters | show_values:
                prms_box_current = prms_box.copy()
                prms_box_current["fc"] = cur["box_color"]

                if show_letters & show_values:
                    text = self.key_to_tex(key, val)
                elif show_letters:
                    text = self.key_to_tex(key)
                else:
                    if cur["type"] == "pi":
                        val += "%"
                    text = "%s" % val

                if show_all | text_params[key]["inferred"]:
                    cur_x = cur["x"]
                    cur_y = cur["y"]

                    if key in nudge_text_pos:
                        if "x" in nudge_text_pos[key]:
                            cur_x += nudge_text_pos[key]["x"]
                        if "y" in nudge_text_pos[key]:
                            cur_y += nudge_text_pos[key]["y"]

                    plt.text(
                        cur_x,
                        cur_y,
                        text,
                        bbox=prms_box_current,
                        color=cur["color"],
                        **kwargs,
                    )
            else:
                pass

        # if self.demo_graph.migrations != []:
        #     plt.text(
        #         rxlim,
        #         max_time,
        #         "Rate\nParameters",
        #         bbox=dict(
        #             boxstyle="round", fc="black", ec="black", alpha=1.0, pad=0.15
        #         ),
        #         color="white",
        #         va="center",
        #         ha="center",
        #     )

        #     plt.axvline(rxlim, linestyle="--", color="black")

        # PLOTTING FOR TIME PARAMS
        values = [self[key].value for key in tau_keys]
        formatted = [pformats["tau"](x) for x in values]

        if show_letters & show_values:
            labels = [
                self.key_to_tex(key, val) for key, val in zip(tau_keys, formatted)
            ]
        elif show_letters:
            labels = [self.key_to_tex(key) for key in tau_keys]
        else:
            labels = ["%s" % val for val in formatted]

        if log_time:
            if values[0] == 0.0:
                values = np.array(values) + 1.0

        if show_letters | show_values:
            ret.set_yticks(
                values, labels, fontsize=tau_font_size
            )  # Show time parameters in yticks
            for i, key in enumerate(tau_keys):
                if show_all | text_params[key]["inferred"]:
                    prms_box_current = deepcopy(prms_box)
                    prms_box_current["fc"] = text_params[key]["box_color"]
                    ret.get_yticklabels()[i].set_color(text_params[key]["color"])
                    ret.get_yticklabels()[i].set_bbox(prms_box_current)
                    ret.get_yticklabels()[i].set_fontsize(tau_font_size)

    def _init_param(
        self,
        x: float,
        key: str,
        demes_event: str,
        param_name: str,
        i: int,
        j: int | None,
        k: int | None,
        fixed: bool = False,
    ):
        """Initialize a new parameter at self[key]. If the key exist it adds deme_dict position to the existing key.

        Args:
            num (int): Numeric value of the parameter
            key (str): key of the parameter in self._Theta
            demes_event (str): 'demes', 'migrations' or 'pulses'
            param_name (str): name of the param. 'start_time', 'end_time' etc.
            i: self[demes_event][i]
            j: self['demes'][i]['epochs'][j]
            k: self[demes_event][i]['proportions'][k]
        """
        if param_name in ["time", "start_time", "end_time"]:
            param_class = TimeParam
        elif param_name in ["end_size", "start_size"]:
            param_class = SizeParam
        elif param_name == "rate":
            param_class = RateParam
        elif param_name == "proportions":
            param_class = ProportionParam
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        if fixed:
            param_class = FixedParam

        path, desc = get_path(self._demo_dict, demes_event, param_name, i, j, k)

        if key in self:
            self[key].add_path(path, desc)
        else:
            val = param_class(
                value=x, path=path, desc=desc, train=param_class is not TimeParam
            )
            # we don't use self[key] = val because it's overridden in this class.
            super().__setitem__(key, val)

    def _repr_html_(self):
        return get_html_repr(self)


class Param:
    """
    Parameter Class. Each params[key] belongs to this class.
    """

    def __init__(
        self,
        value: float,
        lower_bound: float,
        upper_bound: float,
        path: tuple,
        desc: tuple,
        train: bool = False,
    ):
        assert lower_bound <= value <= upper_bound
        self._value = value
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._train = train
        self.paths = {path: desc}

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value: bool):
        self._train = bool(value)

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value):
        assert value <= self.value
        self._lower_bound = value

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value):
        assert value >= self.value
        self._upper_bound = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # assert self.lower_bound <= value <= self.upper_bound
        self._value = jnp.array(value, float)

    def add_path(self, path: tuple, desc: str):
        assert path not in self.paths
        self.paths[path] = desc

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class TimeParam(Param):
    def __init__(self, **kwargs):
        lower_bound = 0.0
        upper_bound = inf
        kwargs.update(lower_bound=lower_bound, upper_bound=upper_bound)
        super().__init__(**kwargs)


class FixedParam(Param):
    def __init__(self, value: float, path: tuple, desc: tuple, train: bool = False):
        self._value = value
        self._upper_bound = self._lower_bound = None
        self._train = False
        self.paths = {path: desc}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        raise ValueError("FixedParam value cannot be changed")

    @property
    def train(self):
        return False

    @train.setter
    def train(self, value):
        raise ValueError("FixedParam cannot be trained")


class SizeParam(Param):
    def __init__(self, **kwargs):
        lower_bound = min(kwargs["value"], 0.01)
        upper_bound = inf
        kwargs.update(lower_bound=lower_bound, upper_bound=upper_bound)
        super().__init__(**kwargs)


class ProportionParam(Param):
    def __init__(self, **kwargs):
        lower_bound = 0.0
        upper_bound = 1.0
        kwargs.update(lower_bound=lower_bound, upper_bound=upper_bound)
        super().__init__(**kwargs)


class RateParam(Param):
    def __init__(self, **kwargs):
        lower_bound = 0.0
        upper_bound = 1.0
        kwargs.update(lower_bound=lower_bound, upper_bound=upper_bound)
        super().__init__(**kwargs)


class LinearConstraints(NamedTuple):
    """Class representing a set of linear constraints on demographic parameters.

    Params:
        keys: The names of the parameters in the constraints.
        constraints: A list of strings representing the constraints.
    """

    keys: list[str]
    constraints: list[str]

    def theta_to_x(self, theta: dict[str, float]) -> jax.Array:
        "Convert theta to x"
        return jnp.array([theta[key] for key in self.keys])

    def x_to_theta(self, x: jax.Array) -> dict[str, float]:
        "Convert x to theta"
        return {key: jnp.array(val).astype(float) for key, val in zip(self.keys, x)}

    def valid(self, theta: dict[str, float], eps: float = 1e-6) -> bool:
        "Check that constraints are satisfiable for given theta"
        A, b, G, h = self.polyhedron
        x = self.theta_to_x(theta)
        return jnp.allclose(A @ x, b) and jnp.all(G @ x <= h + eps)

    def get_projector(self, verbose: bool = False, tol: float = 1e-6):
        "Returns a function which takes a value x, and projects it onto the feasible set"
        poly = self.polyhedron
        f = project_polyhedron(*poly, verbose=verbose, tol=tol)

        def g(params_d):
            x = self.theta_to_x(params_d)
            x_proj = f(x)
            return self.x_to_theta(x_proj)

        g.poly = poly
        return g

    def to_scipy(self, htol=0.0, atol=1e-8, rtol=1e-5):
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
        A, b, G, h = self.polyhedron
        h -= htol
        eps = atol + jnp.abs(b) * rtol
        ret = []
        # scipy does not like constraints with dimension 0
        if A.size > 0:
            ret.append(LinearConstraint(A, b - eps, b + eps))
        if G.size > 0:
            ret.append(LinearConstraint(G, ub=h - htol))
        return ret

    @property
    def polyhedron(self):
        keys = list(self.keys)
        A, b, G, h = [], [], [], []
        for c in self.constraints:
            ret = linear_constraint_vector(c, keys)
            if ret is None:
                # Returns none if there are no parameters in the constraint
                continue
            Ai, bi, operator = ret
            if operator == "EqualTo":
                A.append(Ai)
                b.append(bi)
            else:
                assert operator == "LessThan"
                G.append(Ai)
                h.append(bi)

        def m2n(a):
            return sympy.matrix2numpy(sympy.Matrix(a), dtype=float)

        A = sympy.Matrix(A)
        b = sympy.Matrix(b)
        Abr, _ = A.row_join(b).rref()
        A, b = np.split(m2n(Abr), [-1], axis=1)
        A = A.reshape(-1, len(keys))
        b = b.reshape(-1)
        G = m2n(G).reshape(-1, len(keys))
        h = m2n(h).reshape(-1)
        G, h = _reduce_inequality_constraints(G, h)
        return (A, b, G, h)

    def _pretty_expr(self, Ai, bi, operator):
        # Returns sympy expressions
        keys = self.keys
        str_expr = []
        for coef, key in zip(Ai, keys):
            if np.isclose(coef, 0):
                pass
            elif np.isclose(coef, 1.0):
                str_expr.append(f" {key} ")
            else:
                str_expr.append(" {coef:.3g} * {key}".format(coef=coef, key=key))
        str_expr = "+".join(str_expr)
        if operator == "LessThan":
            operator = "<="
            str_expr += f"{operator}{bi}"
            expr = sympy.simplify(str_expr)
        elif operator == "EqualTo":
            operator = "=="
            str_expr += f"{operator}{bi}"
            lhs, rhs = str_expr.split("==")
            lhs, rhs = sympy.sympify(lhs), sympy.sympify(rhs)
            expr = sympy.Eq(lhs, rhs).simplify()
        else:
            raise ValueError(f"Unknown operator {operator}")
        return expr

    def __repr__(self):
        A, b, G, h = self.polyhedron
        out = []
        for Gi, hi in zip(G, h):
            eq_str = str(self._pretty_expr(Gi, hi, "LessThan"))
            out.append(eq_str)

        for Ai, bi in zip(A, b):
            eq = self._pretty_expr(Ai, bi, "EqualTo")
            lhs, rhs = str(eq.lhs), str(eq.rhs)
            eq_str = lhs + " == " + rhs
            out.append(eq_str)

        return pprint.pformat(out)


def linear_constraint_vector(linear_constraint_str: str, variables: tuple[str]):
    """Takes a string expression and returns Ai, bi and operator.
    Args:
        linear_constraint_str (str): This is an string expression
        variables (list): list of variable names

    Returns:
        Tuple(list, list, list): Ai, bi, Gi, Hi
        where Gi@x<=hi and Ai@x=bi.

    Raises:
        ValueError: If equation is not in correct form for sympy to parse it
    """
    n = len(variables)

    if "==" in linear_constraint_str:
        # Equality constraint
        lhs, rhs = linear_constraint_str.split("==")
        lhs, rhs = sympy.sympify(lhs), sympy.sympify(rhs)
        expr = sympy.Eq(lhs, rhs).simplify()
        operator = "EqualTo"
    elif ("<=" in linear_constraint_str) or (">=" in linear_constraint_str):
        # Inequality constraint
        expr = sympy.simplify(linear_constraint_str)
        operator = expr.__class__.__name__
    else:
        raise ValueError(
            """Equation is not in a correct form:
g(_Theta)<=h(_Theta) or
g(_Theta)>=h(_Theta) or
g(_Theta)==h(_Theta)"""
        )

    # the expression evaluated to a boolean, so it didn't contain
    # any variables. Ensure that the constraint is satisfied and return empty arrays.
    if expr.is_Boolean:
        assert expr
        return None

    lhs = expr.lhs
    rhs = expr.rhs

    is_linear = True
    for x in [rhs, lhs]:
        x = x.as_poly()
        if x is not None:
            is_linear = is_linear & x.is_linear

    if not is_linear:
        raise ValueError("Please provide a linear constraint")

    lhs_params = list(lhs.free_symbols)
    rhs_params = list(rhs.free_symbols)

    lhs_coefs = [lhs.coeff(param) for param in lhs_params]
    rhs_coefs = [rhs.coeff(param) for param in rhs_params]
    lhs_constant = lhs.as_coeff_Add()[0]
    rhs_constant = rhs.as_coeff_Add()[0]

    if operator == "GreaterThan":
        lhs_coefs = [-x for x in lhs_coefs]
        rhs_constant = -rhs_constant
        operator = "LessThan"
    elif operator == "LessThan":
        rhs_coefs = [-x for x in rhs_coefs]
        lhs_constant = -lhs_constant
    else:
        rhs_coefs = [-x for x in rhs_coefs]

    Ai = n * [0]
    non_zero_params = lhs_params + rhs_params
    non_zero_coefs = lhs_coefs + rhs_coefs
    non_zero_indices = [variables.index(str(p)) for p in non_zero_params]
    for i, val in zip(non_zero_indices, non_zero_coefs):
        Ai[i] = val

    bi = float(lhs_constant + rhs_constant)
    return Ai, bi, operator


def set_path(d, path_tup, val):
    """Set value to a nested dictionary.

    Args:
        d (dict): Nested dictionary
        path_tup (tuple): Path to the value
        val (Any): Value to be set
    """
    for key in path_tup[:-1]:
        d = d[key]
    d[path_tup[-1]] = val


def get_path(demo_dict, demes_event, param_name, i, j, k):
    """This returns a path (and its description) to assign values to a demo dict:
    Late it could be used with an update function s.t.
    update(demo_dict, path, value) will change deme_dict[path] = value

    Args:
        demo_dict (dict): demo.asdict() where demo is a demes graph
        demes_event (str): 'demes', 'migrations' or 'pulses'
        param_name (str): name of the param. 'start_time', 'end_time' etc.
        i (int): i; self[demes_event][i]
        j (Union[int, NoneType]): j; self['demes'][i]['epochs'][j]
        k (Union[int, NoneType]): k; self[demes_event][i]['proportions'][k]
    """
    b1 = j is None
    b2 = k is None

    if demes_event == "demes":
        dname = demo_dict[demes_event][i]["name"]
        desc = f"{param_name} of {dname}"
        if not b1:
            desc += f" (epoch {j})"
    elif demes_event == "pulses":
        sources = " ".join(demo_dict[demes_event][i]["sources"])
        dest = demo_dict[demes_event][i]["dest"]
        desc = f"{param_name} of the pulse from {sources} to {dest}"
    elif demes_event == "migrations":
        source = demo_dict[demes_event][i]["source"]
        dest = demo_dict[demes_event][i]["dest"]
        desc = f"{param_name} of the migration from {source} to {dest}"
    else:
        raise ValueError(f"Unknown {demes_event=}")

    if b1 & b2:
        path = (demes_event, i, param_name)
    elif not b1:
        # epoch var assignment
        path = (demes_event, i, "epochs", j, param_name)
    else:
        # proportion assignment
        path = (demes_event, i, param_name, k)

    return path, desc


def _reduce_inequality_constraints(G, h):
    """Remove redundant inequalities from a system of linear inequalities G @ x0 <= h.

    Args:
        G (2d array): Coefficients of the inequalities
        h (1d array): Constants of the inequalities

    Returns:
        Reduced G, h.
    """
    i = 0
    while i < len(G):
        # len(G) could change from the previous iteration
        Id = np.eye(len(G))
        e = Id[i]
        hi = h + e

        # If the constraint is redundant, then the optimal value of the
        # following linear program should be greater than or equal to h[i]:

        # **** Don't change method="highs" to "simplex", simplex will fail to find
        # the optimal solution in some cases, which results in constraints being
        # silently dropped!
        res = linprog(-G[i], G, hi, bounds=(None, None), method="highs")

        if -res.fun <= h[i]:
            # constraint is redundant
            # logger.debug("constraint {}<={} is redundant", G[i], h[i])
            G = np.delete(G, i, axis=0)
            h = np.delete(h, i, axis=0)
        else:
            i += 1
    return np.copy(G), np.copy(h)  # return copies


def get_body(vals, styles):
    ret = ""
    for row in vals:
        ret += "<tr>\n"
        for j, ri in enumerate(row):
            style = styles[j]
            ret += f'<td style="{style}">{ri}</td>\n'
        ret += "</tr>\n"
    return ret


def get_html_repr(params):
    # Returns html representation of params

    def eq_to_html(eq):
        eq = re.sub(r"_(\d+)", r"<sub>\1</sub>", eq)
        eq = eq.replace("<=", "≤")
        eq = eq.replace(">=", "≥")
        eq = eq.replace("==", "=")
        for key, value in Greek.items():
            eq = eq.replace(key, value)
        return eq

    body = {
        "table1": {
            "RateParam": "",
            "TimeParam": "",
            "SizeParam": "",
            "ProportionParam": "",
        },
        "table2": "",
    }
    param_types = [
        "SizeParam",
        "RateParam",
        "ProportionParam",
        "TimeParam",
        "FixedParam",
    ]
    Greek = dict(zip(["eta", "rho", "pi", "tau"], ["&#951", "&#961", "&#960", "&#964"]))
    keys = list(params.keys())

    body = {"table1": {}}

    styles = {
        "table1": [
            "text-align:left; font-family:'Lucida Console'",
            "font-family:'Lucida Console'",
            "",
        ],
        "table2": [
            "text-align:left; font-family:'Lucida Console', monospace",
            "font-family:'Lucida Console'",
        ],
        "table_eq": ["text-align:left; font-family:'Lucida Console', monospace"],
    }

    table2_vals = []
    for param_type in param_types:
        table1_vals = []
        for key in keys:
            cur = params[key]
            if cur.__class__.__name__ == param_type:
                i = int(re.findall(r"\d+", key)[0])
                key_greek = re.findall(r"[a-z]+", key)[0]
                name = Greek[key_greek] + f"<sub>{i}</sub>"
                num = "{num:.3g}".format(num=cur.value)
                a = [name, num]
                if cur.__class__ is not FixedParam:
                    if params[key].train:
                        train = "&#9989"
                    else:
                        train = "&#10060"
                    a.append(train)
                table1_vals.append(a)
                for desc in cur.paths.values():
                    table2_vals.append([desc, name])
        table1_vals = sorted(table1_vals, key=lambda x: x[0])
        body["table1"][param_type] = get_body(table1_vals, styles=styles["table1"])

        table2_vals = sorted(table2_vals, key=lambda x: x[1])
    body["table2"] = get_body(table2_vals, styles=styles["table2"])

    eq_table = []
    # for eq in params.constraints.user_constraints.keys():
    #     eq_table.append([eq_to_html(eq)])
    eq_table = get_body(eq_table, styles=styles["table_eq"])

    # FIXME: Add the actual link to paper
    return f"""
<div style="display: inline-block; width: 30%;">
    <a href="https://github.com/jthlab/momi3" target="_blank">SOURCE CODE</a>
    <a href="https://www.biorxiv.org/content/10.1101/2024.03.26.586844v1" target="_blank">PAPER</a>
    <br>
    <img src="https://enesdilber.github.io/momilogo.png" style="width:75px;height:52px;">
    <table border="1" style="width: 100%;">
    <caption><h4>Size Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body['table1']['SizeParam']}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Rate Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body['table1']['RateParam']}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Proportion Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body['table1']['ProportionParam']}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Time Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body['table1']['TimeParam']}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Fixed Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
        </tr>
    </thead>
    <tbody>
    {body['table1']['FixedParam']}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Constraints</h4></caption>
    <thead>
        <tr style="text-align: left;">
            <th style="text-align:left;">User Constraints</th>
        </tr>
    </thead>
    <tbody>
    {eq_table}
    </tbody>
    </table>
</div>
<div style="display: inline-block; width: 50%;">
<br>
    <table border="1" style="width: 100%;">
    <caption><h4>Parameter Locations</h4></caption>
    <thead>
        <tr>
            <th style="text-align:left; width:80%">Demes Parameter</th>
            <th >Parameter</th>
        </tr>
    </thead>
    <tbody>
    {body['table2']}
    </tbody>
    </table>
</div>
"""
