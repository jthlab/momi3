from __future__ import annotations

import math
import re
from copy import deepcopy
from math import inf, isinf
from typing import Callable, Union

import demes
import demesdraw
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import LinearConstraint, linprog

from momi3.utils import signif, update

NoneType = type(None)
NP_max = np.finfo(np.float32).max
EPS_time_differences = 0.01  # to avoid 0 division, =min(t1-t0)


def Intergers():
    i = 0
    while True:
        yield i
        i += 1


class Params(dict):

    """Parameter class.

    Attributes:
        add_linear_constraint (func): Adds a new user constraint
        set (func): set a new value for the given parameter
        set_train (func): set to learn the optimum parameter value
    """

    def __init__(self, momi=None, demo_dict=None, T=None):
        if momi is not None:
            T = momi._T
            demo_dict = momi._demo_dict
        demo_dict = deepcopy(demo_dict)
        self._demo_dict = demo_dict
        self._T = T
        self._frozen = False
        self._paths_to_params = {}
        self._path_to_params = {}
        self._params_to_paths = {}

        # Create time keys dictionary: tau_0, tau_1, ..., tau_p.
        # time_0 < time_1 < ... < time_p
        # tkeys[time_i] = tau_i
        ts = set(float(i.t.t) for i in T.nodes())
        ts = sorted(list(ts))
        tkeys = dict(zip(ts, [f"tau_{i}" for i in range(len(ts))]))

        # iter size, rate and prop parameters.
        # Note: Not iterating time parameters because same time points have the same time key (Generated above).
        iter_size = Intergers()
        iter_rate = Intergers()
        iter_prop = Intergers()

        # 1 Demes:
        demes_event = "demes"
        for i, deme in enumerate(self._demo_dict["demes"]):
            j = None
            k = None

            # Start time of a Deme:
            param_name = "start_time"
            num = float(deme[param_name])
            key = tkeys[num]
            self._init_Theta(num, key, demes_event, param_name, i, j, k)

            # Proportion of Ancestors:
            param_name = "proportions"
            if len(deme["proportions"]) == 1:
                # Single ancestor, proportion is fixed to 1.
                pass
            else:
                for k, proportion in enumerate(deme["proportions"]):
                    num = float(proportion)
                    key = f"pi_{next(iter_prop)}"
                    self._init_Theta(num, key, demes_event, param_name, i, j, k)

            # Iterate epochs
            for j, epoch in enumerate(deme["epochs"]):
                # End time of an Epoch:
                param_name = "end_time"
                num = float(epoch[param_name])
                key = tkeys[num]
                self._init_Theta(num, key, demes_event, param_name, i, j, k)

                # Size of an Epoch:
                if epoch["size_function"] == "constant":
                    val = next(iter_size)
                    keys = 2 * [f"eta_{val}"]
                else:
                    # Exponential pop size, default _Theta key
                    keys = [f"eta_{next(iter_size)}", f"eta_{next(iter_size)}"]
                for key, param_name in zip(keys, ["start_size", "end_size"]):
                    num = float(epoch[param_name])
                    self._init_Theta(num, key, demes_event, param_name, i, j, k)

        # 2 Migrations:
        j = None
        k = None
        migration_rates = {}
        demes_event = "migrations"
        for i, migration in enumerate(self._demo_dict["migrations"]):
            # Time of a Migration:
            for param_name in ["start_time", "end_time"]:
                num = float(migration[param_name])
                key = tkeys[num]
                self._init_Theta(num, key, demes_event, param_name, i, j, k)

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
            num = float(migration[param_name])
            self._init_Theta(num, key, demes_event, param_name, i, j, k)

        # 3 Pulses:
        demes_event = "pulses"
        for i, pulse in enumerate(self._demo_dict["pulses"]):
            j = None
            k = None

            # Time of the pulse
            param_name = "time"
            num = float(pulse[param_name])
            key = tkeys[num]
            self._init_Theta(num, key, demes_event, param_name, i, j, k)

            # Proportions of the pulse
            param_name = "proportions"
            for k, proportion in enumerate(pulse["proportions"]):
                num = float(proportion)
                key = f"pi_{next(iter_prop)}"
                self._init_Theta(num, key, demes_event, param_name, i, j, k)
                pulse[param_name][k] = num

        # Linear time constraints for times
        time_constraints_str_exprs = set([])
        for t0, t1 in T.edges():
            i = t0.t.t
            j = t1.t.t
            if all([i != j, not isinf(i), not isinf(j)]):
                time_constraints_str_exprs.add(
                    f"{tkeys[i]}<={tkeys[j]}-{EPS_time_differences}"
                )
        self._linear_constraints = LinearConstraints(self, time_constraints_str_exprs)

        # paths to params keys
        for key in self:
            paths = tuple(self[key].paths)
            self._paths_to_params[paths] = key
            self._params_to_paths[key] = paths
            for path in paths:
                self._path_to_params[path] = key

        self._frozen = True

    def set_train(self, key: str, value: bool):
        self[key].train(value)

    def set_train_all_etas(self, value: bool):
        keys = self._keys
        [self[key].train(value) for key in keys if isinstance(self[key], SizeParam)]

    def set_train_all_rhos(self, value: bool):
        keys = self._keys
        [self[key].train(value) for key in keys if isinstance(self[key], RateParam)]

    def set_train_all_pis(self, value: bool):
        keys = self._keys
        [
            self[key].train(value)
            for key in keys
            if isinstance(self[key], ProportionParam)
        ]

    def set_train_all_taus(self, value: bool):
        keys = self._keys
        [self[key].train(value) for key in keys if isinstance(self[key], TimeParam)]

    def set(self, key: str, value: float):
        value = float(value)
        self._check_parameter(key)
        x = self._theta
        self._linear_constraints.check_assignment(key, value, x)
        self[key].set(value)
        for path in self[key].paths:
            # change the values in demo_dict too
            value = float(value)
            self._demo_dict = update(self._demo_dict, path, value)

    def _is_theta_train_valid(self, theta_train_hat: dict | jnp.ndarray) -> bool:
        keys = self._train_keys
        if isinstance(theta_train_hat, dict):
            theta_train_hat = jnp.array(
                [theta_train_hat[key] for key in keys], dtype="f"
            )
        A, b, G, h = self._polyhedron_hyperparams()
        b1 = jnp.all(G @ theta_train_hat <= h)
        b2 = jnp.allclose(A @ theta_train_hat, b)
        return b1 & b2

    def set_optimization_results(self, theta_train_hat: dict | jnp.ndarray):
        keys = self._train_keys
        if isinstance(theta_train_hat, dict):
            theta_train_hat = jnp.array(
                [theta_train_hat[key] for key in keys], dtype="f"
            )

        assert len(keys) == len(theta_train_hat)

        if self._is_theta_train_valid(theta_train_hat):
            pass
        else:
            raise ValueError("Invalid theta_train_hat")

        for key, value in zip(keys, theta_train_hat):
            value = float(value)
            self[key].set(value)
            for path in self[key].paths:
                # change the values in demo_dict too
                self._demo_dict = update(self._demo_dict, path, value)

    def add_linear_constraint(self, expr_str):
        x = self._theta
        self._linear_constraints.add_constraint(expr_str, x)

    @property
    def _keys(self):
        return sorted(list(self.keys()))

    @property
    def _theta(self):
        keys = self._keys
        return [self[key].num for key in keys]

    @property
    def _train_bool(self):
        keys = self._keys
        return [self[key].train_it for key in keys]

    @property
    def _train_keys(self):
        keys = self._keys
        bools = self._train_bool
        return [key for key, b in zip(keys, bools) if b]

    @property
    def _nuisance_keys(self):
        keys = self._keys
        bools = self._train_bool
        return [key for key, b in zip(keys, bools) if not b]

    @property
    def _theta_train(self):
        keys = self._keys
        bools = self._train_bool
        return [self[key].num for key, b in zip(keys, bools) if b]

    @property
    def _theta_nuisance(self):
        keys = self._keys
        bools = self._train_bool
        return [self[key].num for key, b in zip(keys, bools) if not b]

    @property
    def _theta_train_dict(self):
        keys = self._keys
        bools = self._train_bool
        paths_train = [tuple(self[key].paths) for key, b in zip(keys, bools) if b]
        theta_train = self._theta_train
        return dict(zip(paths_train, theta_train))

    @property
    def _theta_nuisance_dict(self):
        keys = self._keys
        bools = self._train_bool
        paths_nuisance = [
            tuple(self[key].paths) for key, b in zip(keys, bools) if not b
        ]
        theta_nuisance = self._theta_nuisance
        return dict(zip(paths_nuisance, theta_nuisance))

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
        if ptype in ['tau', 'eta']:
            # it's actually log differences of tau
            if inverse:
                ret = math.exp(val)
            else:
                ret = math.log(val)

        elif ptype in ['rho', 'pi']:
            if inverse:
                ret = 1 / (1 + math.exp(-val))
            else:
                ret = math.log(val / (1 - val))

        else:
            raise ValueError(f'Unknown {ptype=}')

        return ret

    @property
    def _transformed_diff_tau_dict(self) -> tuple[dict[tuple, float], dict[tuple, float]]:
        # returns infer and no inter keys
        # This dict stores log(tau[i] - tau[i-1])

        ptype = 'tau'
        keys = self._keys
        tau_keys = [key for key in keys if isinstance(self[key], TimeParam)]
        tau_keys = sorted(tau_keys, key=lambda key: self[key].num)
        tau_vals = [self[key].num for key in tau_keys]
        tau_infr = [self[key].train_it for key in tau_keys]

        n_diff_tau = len(tau_keys) - 1

        diff_tau_vals = [tau_vals[i + 1] - tau_vals[i] for i in range(n_diff_tau)]
        diff_tau_infr = [tau_infr[i + 1] | tau_infr[i] for i in range(n_diff_tau)]
        trans_tau_vals = [self.transform_fns(val, ptype) for val in diff_tau_vals]
        trans_tau_paths = [
            (
                self._params_to_paths[tau_keys[i + 1]],
                self._params_to_paths[tau_keys[i]]
            ) for i in range(len(tau_keys) - 1)
        ]

        diff_tau_train_dict = {
            trans_tau_paths[i][1]: {
                trans_tau_paths[i][0]: trans_tau_vals[i]
            } for i in range(n_diff_tau) if diff_tau_infr[i]
        }

        diff_tau_nuisance_dict = {
            trans_tau_paths[i][1]: {
                trans_tau_paths[i][0]: trans_tau_vals[i]
            } for i in range(n_diff_tau) if not diff_tau_infr[i]
        }

        diff_tau_nuisance_dict[(('init', ),)] = {self._params_to_paths['tau_0']: tau_vals[0]}

        return diff_tau_train_dict, diff_tau_nuisance_dict

    @property
    def _transformed_rho_dict(self):
        ptype = 'rho'
        cur_keys = [key for key in self if isinstance(self[key], RateParam)]
        _train_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if self[key].train_it}
        _nuisance_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if not self[key].train_it}
        return _train_dict, _nuisance_dict

    @property
    def _transformed_pi_dict(self):
        ptype = 'pi'
        cur_keys = [key for key in self if isinstance(self[key], ProportionParam)]
        _train_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if self[key].train_it}
        _nuisance_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if not self[key].train_it}
        return _train_dict, _nuisance_dict

    @property
    def _transformed_eta_dict(self):
        ptype = 'eta'
        cur_keys = [key for key in self if isinstance(self[key], SizeParam)]
        _train_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if self[key].train_it}
        _nuisance_dict = {tuple(self[key].paths): self.transform_fns(self[key].num, ptype) for key in cur_keys if not self[key].train_it}
        return _train_dict, _nuisance_dict

    @property
    def _transformed_theta_train_dict(self):
        transformed_theta_train_dict = []
        for td, nd in [
            self._transformed_eta_dict,
            self._transformed_rho_dict,
            self._transformed_pi_dict,
            self._transformed_diff_tau_dict
        ]:
            transformed_theta_train_dict.append(td)
        return tuple(transformed_theta_train_dict)

    @property
    def _transformed_theta_nuisance_dict(self):
        transformed_theta_nuisance_dict = []
        for td, nd in [
            self._transformed_eta_dict,
            self._transformed_rho_dict,
            self._transformed_pi_dict,
            self._transformed_diff_tau_dict
        ]:
            transformed_theta_nuisance_dict.append(nd)
        return tuple(transformed_theta_nuisance_dict)

    def _polyhedron_hyperparams(self, htol=0.0):
        # See: https://jaxopt.github.io/stable/_autosummary/jaxopt.projection.projection_polyhedron.html
        A, b, G, h = self._linear_constraints.get_polyhedron_hyperparams(
            self._theta, self._train_bool
        )
        return A, b, G, h - htol

    def _linear_constraints_for_scipy(self, htol=0.0, atol=1e-8, rtol=1e-5):
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html
        A, b, G, h = self._polyhedron_hyperparams(htol)
        eps = atol + jnp.abs(b) * rtol
        LCs = [LinearConstraint(A, b - eps, b + eps), LinearConstraint(G, ub=h - htol)]
        return LCs

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
            ret = r"$%s=%s$" % (key, val)
        return ret

    def _solve_y_conflict(self, text_params, log_time):
        positions = set([])
        for key in text_params:
            cur = text_params[key]
            ymin, ymax = cur['ymin'], cur['ymax']
            r = ymax - ymin
            i = 2
            cont = True
            while cont:
                for j in range(1, i):
                    if log_time:
                        position = 1 + ymin + r * (10 ** j) / (10 ** i)
                    else:
                        position = ymin + r * j / i
                    if positions.issuperset({position}):
                        pass
                    else:
                        cont = False
                        positions.add(position)
                        break
                i += 1
            cur['y'] = position
            del cur['ymin']
            del cur['ymax']

    def tubes(
        self,
        show_values: bool = True,
        show_letters: bool = False,
        USER_DICT: dict[str, float] = None,
        color_intensity_function: Callable = lambda x: x,
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
        dG = self.demo_graph
        ret = demesdraw.tubes(dG, **kwargs)

        lxlim, rxlim = ret.get_xlim()
        min_time, max_time = ret.get_ylim()

        log_time = kwargs.get("log_time", False)

        if USER_DICT is not None:
            box_color_by = "USER_DICT"
        else:
            box_color_by = "train"

        default_kwargs = {"kwargs": {"va": "bottom", "ha": "center"}}

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
                        key = self._path_to_params[path]
                        text_params[key] = {
                            "type": "pi",
                            "x": x,
                            "y": y,
                            "inferred": key in self._train_keys,
                        }
                        text_params[key].update(default_kwargs)

                        text = signif(dG.pulses[j].proportions[0])
                        pulse_text.append({"x": x, "y": y, "text": text})
                        break

        # MIGRATIONS
        times = [i.time for i in dG.pulses]
        pulse_text = []
        for i, line in enumerate(ret.get_lines()):
            x = line._x
            y = line._y
            if len(y) == 2:
                for j, t in enumerate(times):
                    if jnp.isclose(t, y[0]):
                        x = (x[0] + x[1]) / 2
                        y = y[0]
                        y = np.where((y < 1) & log_time, 1 + y, y)

                        path = ("pulses", j, "proportions", 0)
                        key = self._path_to_params[path]
                        text_params[key] = {
                            "type": "pi",
                            "x": x,
                            "y": y,
                            "inferred": key in self._train_keys,
                        }
                        text_params[key].update(default_kwargs)

                        text = signif(dG.pulses[j].proportions[0])
                        pulse_text.append({"x": x, "y": y, "text": text})
                        break

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
                    text = signif(texts[1])
                    y = ys[1]
                    y = np.where((y < 1) & log_time, 1 + y, y)

                    path = ("demes", i, "epochs", j, "end_size")
                    key = self._path_to_params[path]
                    text_params[key] = {
                        "type": "eta",
                        "x": x,
                        "y": y,
                        "inferred": key in self._train_keys,
                    }
                    text_params[key].update(default_kwargs)

                else:
                    # exponential growth
                    var_type = iter(["start_size", "end_size"])
                    if not ys[0] > max_time:
                        for k in range(2):
                            text = signif(texts[k])
                            y = ys[k]
                            y = np.where((y < 1) & log_time, 1 + y, y)
                            # y = np.clip(y, a_min=step, a_max=max_time)
                            if k == 0:
                                va = "top"
                            else:
                                va = "bottom"

                            path = ("demes", i, "epochs", j, next(var_type))
                            key = self._path_to_params[path]
                            text_params[key] = {
                                "type": "eta",
                                "x": x,
                                "y": y,
                                "inferred": key in self._train_keys,
                            }
                            text_params[key].update(default_kwargs)
                            text_params[key] = deepcopy(text_params[key])
                            text_params[key]["kwargs"].update({"va": va})

        rho_keys = sorted([key for key in self if (key[:3] == "rho")])
        # MIGRATIONS
        mig_params = {}
        for key in rho_keys:
            mig_params[key] = {"inferred": self[key].train_it}
            val = signif(self[key].num)
            mig_path = list(list(self[key].paths)[0])
            start_time = mig_path[:-1] + ["start_time"]
            end_time = mig_path[:-1] + ["end_time"]
            st_key = self._path_to_params[tuple(start_time)]
            en_key = self._path_to_params[tuple(end_time)]
            mig_params[key] = {
                "type": "rho",
                "x": rxlim,
                "ymin": self[en_key].num,
                "ymax": self[st_key].num,
                "inferred": key in self._train_keys,
            }
            mig_params[key].update(default_kwargs)
            mig_params[key] = deepcopy(mig_params[key])

        self._solve_y_conflict(mig_params, log_time)
        text_params.update(mig_params)

        tau_keys = sorted(
            [key for key in self if (key[:3] == "tau") & (not isinf(self[key].num))]
        )

        non_tau_keys = sorted([key for key in self if key[:3] != "tau"])

        for key in tau_keys:
            text_params[key] = {"inferred": self[key].train_it}

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
            val = signif(self[key].num)
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
                    text = "%s" % val

                plt.text(
                    cur["x"],
                    cur["y"],
                    text,
                    bbox=prms_box_current,
                    color=cur["color"],
                    **kwargs,
                )
            else:
                pass

        if self.demo_graph.migrations != []:
            plt.text(
                rxlim,
                max_time,
                "Rate\nParameters",
                bbox=dict(
                    boxstyle="round", fc="black", ec="black", alpha=1.0, pad=0.15
                ),
                color="white",
                va="center",
                ha="center",
            )

            plt.axvline(rxlim, linestyle="--", color="black")

        # PLOTTING FOR TIME PARAMS
        values = [self[key].num for key in tau_keys]

        if show_letters & show_values:
            labels = [
                self.key_to_tex(key, signif(val)) for key, val in zip(tau_keys, values)
            ]
        elif show_letters:
            labels = [self.key_to_tex(key) for key in tau_keys]
        else:
            labels = ["%s" % signif(val) for val in values]

        if log_time:
            if values[0] == 0.:
                values = np.array(values) + 1.0
        ret.set_yticks(values, labels)  # Show time parameters in yticks

        if show_letters | show_values:
            for i, key in enumerate(tau_keys):
                prms_box_current = deepcopy(prms_box)
                prms_box_current["fc"] = text_params[key]["box_color"]
                ret.get_yticklabels()[i].set_color(text_params[key]["color"])
                ret.get_yticklabels()[i].set_bbox(prms_box_current)

    def _init_Theta(
        self,
        num: float,
        key: str,
        demes_event: str,
        param_name: str,
        i: int,
        j: Union[int, NoneType],
        k: Union[int, NoneType],
    ):
        """Initiates self[key]. If the key exist it adds deme_dict position to
        the existing key.
        Args:
            num (int): Numeric value of the parameter
            key (str): key of the parameter in self._Theta
            demes_event (str): 'demes', 'migrations' or 'pulses'
            param_name (str): name of the param. 'start_time', 'end_time' etc.
            i (int): i; self[demes_event][i]
            j (Union[int, NoneType]): j; self['demes'][i]['epochs'][j]
            k (Union[int, NoneType]): k; self[demes_event][i]['proportions'][k]
        Returns:
            demoParam
        """

        if param_name in ["time", "start_time", "end_time"]:
            if isinf(num):
                param_class = RootTimeParam
            else:
                param_class = TimeParam
        elif param_name in ["end_size", "start_size"]:
            param_class = SizeParam
        elif param_name == "rate":
            param_class = RateParam
        elif param_name == "proportions":
            param_class = ProportionParam
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        path, demes_params_desc = get_path(
            self._demo_dict, demes_event, param_name, i, j, k
        )

        if key in self:
            self[key].add_param(path, demes_params_desc)
        else:
            self[key] = param_class(
                num=num, path=path, demes_params_desc=demes_params_desc
            )

        return None

    def _repr_html_(self):
        return get_html_repr(self)

    def __setitem__(self, key, value):
        if self._frozen:
            self.set(key, value)
        else:
            super().__setitem__(key, value)

    def copy(self):
        params_copy = Params(demo_dict=self.demo_dict, T=self._T)
        train_bool = self._train_bool
        keys = self._keys
        [params_copy.set_train(key, value) for key, value in zip(keys, train_bool)]
        params_copy._linear_constraints.user_constraint_dict = deepcopy(
            self._linear_constraints.user_constraint_dict
        )
        return params_copy


class Param(object):
    """
    Parameter Class. Each params[key] is belong to this class.
    """

    _frozen = False
    _accepted_keys = ["num", "LB", "UB", "demes_params_descs", "paths", "_frozen"]

    def __init__(
        self, num: float, LB: float, UB: float, demes_params_desc: str, path: tuple
    ):
        self.num = num
        self.UB = UB
        self.LB = LB
        self.demes_params_descs = set([demes_params_desc])
        self.paths = set([path])
        self.train_it = False
        self._frozen = True

    def __setattr__(self, key, value):
        if self._frozen & (key in self._accepted_keys[3:]):
            raise ValueError(f"{key} cannot be changed")
        elif self._frozen & (key == "num"):
            raise ValueError(f"Use method set({value})")
        else:
            object.__setattr__(self, key, value)

    def set(self, value: Union[float, Param]):
        if isinstance(value, Param):
            value = value.num
        if any([value > self.UB, value < self.LB]):
            raise ValueError(f"Assignment should be in [{self.LB}, {self.UB}]")
        else:
            object.__setattr__(self, "num", value)

    def set_LB(self, value: float):
        object.__setattr__(self, "LB", value)

    def set_UB(self, value: float):
        object.__setattr__(self, "UB", value)

    def train(self, train_it: bool):
        self.train_it = train_it

    def add_param(self, path, demes_params_desc):
        self.demes_params_descs.add(demes_params_desc)
        self.paths.add(path)

    def __str__(self):
        return str(self.num)

    def __repr__(self):
        return str(self)


class TimeParam(Param):
    def __init__(self, **kwargs):
        LB = 0.0
        UB = inf
        kwargs.update(LB=LB, UB=UB)
        super().__init__(**kwargs)


class RootTimeParam(Param):
    def __init__(self, **kwargs):
        LB = inf
        UB = inf
        kwargs.update(LB=LB, UB=UB)
        super().__init__(**kwargs)

    def train(self, train_it: bool):
        if train_it:
            raise ValueError(
                "This is the start time of the root time and equal to infinity. Untrainable."
            )
        else:
            pass


class SizeParam(Param):
    def __init__(self, **kwargs):
        LB = 0.01
        UB = inf
        kwargs.update(LB=LB, UB=UB)
        super().__init__(**kwargs)


class ProportionParam(Param):
    def __init__(self, **kwargs):
        LB = 0.0
        UB = 1.0
        kwargs.update(LB=LB, UB=UB)
        super().__init__(**kwargs)


class RateParam(Param):
    def __init__(self, **kwargs):
        LB = 0.0
        UB = 1.0
        kwargs.update(LB=LB, UB=UB)
        super().__init__(**kwargs)


class LinearConstraints(object):

    """Handles Linear constraints of the momi model

    Attributes:
        add_constraint (func): Adds new user constraint
        all_constraints (dict): Returns all constraints
        check_assignment (func): Check if the assignment violates any constraint
        check_constraint (func): Check if the constraint is valid
        check_equation (func): Check if the equation is valid
        discard_constraint (func): Discard the given user constraint
        get_train_arrays (func): Returns projection_polyhedron arrays
        hard_constraints (dict): Necessary constraints for the model
        keys (list): Variable names. e.g. eta_0, eta_1, ...
        user_constraint_dict (dict): User defined constraints
    """

    def __init__(self, params, time_constraints_str_exprs):
        self.keys = params._keys
        self.hard_constraints = {}
        self.user_constraint_dict = dict()

        # Initiate hard constraints:
        time_lc_vecs = {"A": [], "b": []}
        other_lc_vecs = {"A": [], "b": []}
        for key in self.keys:
            cur_param = params[key]
            A = []
            b = []

            LB = cur_param.LB
            if not isinf(LB):
                Ai, bi, operator = linear_constraint_vector(f"{key}>={LB}", self.keys)
                A.append(np.array(Ai, dtype="f"))
                b.append(bi)

            UB = cur_param.UB
            if not isinf(UB):
                Ai, bi, operator = linear_constraint_vector(f"{key}<={UB}", self.keys)
                A.append(np.array(Ai, dtype="f"))
                b.append(bi)

            if cur_param.__class__.__name__ == "TimeParam":
                time_lc_vecs["A"] += A
                time_lc_vecs["b"] += b
            else:
                other_lc_vecs["A"] += A
                other_lc_vecs["b"] += b

        for str_expr in time_constraints_str_exprs:
            Ai, bi, operator = linear_constraint_vector(str_expr, self.keys)
            time_lc_vecs["A"].append(np.array(Ai, dtype="f"))
            time_lc_vecs["b"].append(bi)

        A_time = np.array(time_lc_vecs["A"], dtype="f")
        b_time = np.array(time_lc_vecs["b"])
        A_time, b_time = reduce_linear_constraints(A_time, b_time)
        time_lc_vecs["A"] = list(A_time)
        time_lc_vecs["b"] = list(b_time)

        self.hard_constraints = {
            "A": time_lc_vecs["A"] + other_lc_vecs["A"],
            "b": time_lc_vecs["b"] + other_lc_vecs["b"],
        }

    def check_assignment(self, key, value, x):
        """Check if the assignment violates any constraint.

        Args:
            key (str): Parameter name e.g. "eta_2"
            value (float): Parameter value
            x (list[float]): Values of all paremeters. Ordering is in self.keys
        """
        keys = self.keys
        x[[i for (i, key_i) in enumerate(keys) if key_i == key][0]] = value
        x = np.array(x, dtype="f")
        hard_constraints = self.hard_constraints

        operator = "LessThan"
        for i in range(len(hard_constraints["b"])):
            Ai = hard_constraints["A"][i]
            bi = hard_constraints["b"][i]
            cond = self.check_equation(Ai, x, bi, operator)
            if not cond:
                self._raise_constraint_error(Ai, bi, operator)
            else:
                pass

    def check_constraint(self, expr_str, x):
        """Check if the constraint is valid.

        Args:
            expr_str (str): It should be a valid string expression
            x (list[float]): Values of all paremeters. Ordering is in self.keys
        """
        keys = self.keys
        x = np.array(x, dtype="f")
        Ai, bi, operator = linear_constraint_vector(expr_str, keys)
        Ai = np.array(Ai, dtype="f")
        cond = self.check_equation(Ai, x, bi, operator)
        if not cond:
            raise ValueError("Constraints should hold when you are assigning them")
        else:
            pass

    def check_equation(self, Ai, x0, bi, operator):
        """Check if the equation is valid.
        True if A \times x0^T <= bi or A \times x0^T == bi

        Args:
            Ai (array): Coefficients of parameters in self.keys
            x0 (array): Values of parameters in self.keys
            bi (float): Right hand side of the equation
            operator (str): 'LessThan' or EqualTo

        Returns:
            bool: Is equation valid?
        """
        x0 = self._safe_inf(x0)
        if operator == "EqualTo":
            return np.isclose(Ai @ x0, bi)
        elif operator == "LessThan":
            return Ai @ x0 <= bi
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def add_constraint(self, expr_str, x):
        """Adds a user constraint if the constraint is valid for x.

        Args:
            expr_str (str): It should be a valid string expression
            x (list[float]): Values of all paremeters. Ordering is in self.keys
        """
        keys = self.keys
        self.check_constraint(expr_str, x)  # Test whether it is a valid constraint
        Ai, bi, operator = linear_constraint_vector(expr_str, keys)
        self.user_constraint_dict[expr_str] = {
            "Ai": np.array(Ai, dtype="f"),
            "bi": bi,
            "operator": operator,
        }

    def discard_constraint(self, expr_str):
        """Discards a user constraint if the constraint is available.

        Args:
            expr_str (str): It should be a valid string expression
        """
        del self.user_constraint_dict[expr_str]

    @property
    def all_constraints(self):
        """Returns all constraints.
        ret['LessThan'] is for A @ x <= b
        ret['EqualTo'] is for A @ x == b

        Returns:
            dict: Constraint matrices.
        """
        hard_constraints = deepcopy(self.hard_constraints)

        # All hard constraints have format: g(Theta) <= b
        ret = {
            "LessThan": {"A": hard_constraints["A"], "b": hard_constraints["b"]},
            "EqualTo": {"A": [], "b": []},
        }

        for cur_constraint in self.user_constraint_dict.values():
            operator = cur_constraint["operator"]
            ret[operator]["A"].append(cur_constraint["Ai"])
            ret[operator]["b"].append(cur_constraint["bi"])

        return ret

    def get_polyhedron_hyperparams(self, theta, train_bool):
        """Returns projection_polyhedron arrays.
        See: https://jaxopt.github.io/stable/_autosummary/jaxopt.projection.projection_polyhedron.html

        Args:
            theta (list[float]): Values of all paremeters. Ordering is in self.keys
            train_bool (list[bool]): Boolean for inference. Ordering is in self.keys

        Returns:
            (array, array, array, array): G, h, A, b s.t. Gx <= h and Ax == b
        """

        all_constraints = self.all_constraints
        x = self._safe_inf(np.array(theta, dtype="f"))

        G = np.array(all_constraints["LessThan"]["A"], dtype="f")
        h = np.array(all_constraints["LessThan"]["b"], dtype="f")

        A = np.array(all_constraints["EqualTo"]["A"], dtype="f")
        b = np.array(all_constraints["EqualTo"]["b"], dtype="f")

        x_nt = x[np.logical_not(train_bool)]

        G, G_nt = G[:, train_bool], G[:, np.logical_not(train_bool)]
        h = h - G_nt @ x_nt

        nonzero = np.logical_not(np.alltrue(np.isclose(G, 0.0), 1))
        G = G[nonzero, :]
        h = h[nonzero]

        if len(b) != 0:
            A, A_nt = A[:, train_bool], A[:, np.logical_not(train_bool)]
            b = b - A_nt @ x_nt

            nonzero = np.logical_not(np.alltrue(np.isclose(A, 0.0), 1))
            A = A[nonzero, :]
            b = b[nonzero]
        else:
            A = jnp.zeros(G.shape[1])[None, :]
            b = jnp.array([0.0])

        return (
            jnp.array(A, dtype="f"),
            jnp.array(b, dtype="f"),
            jnp.array(G, dtype="f"),
            jnp.array(h, dtype="f"),
        )

    def get_bounds(self, theta, train_bool):
        """Returns list of [lower_bound, upper_bound] for train=True parameters.
        It maximizes (and minimizes for the lower bound) the value of the parameter
        for given constraints in params._linear_constraints

        Args:
            theta (list[float]): Values of all paremeters. Ordering is in self.keys
            train_bool (list[bool]): Boolean for inference. Ordering is in self.keys

        Returns:
            List[List[float, float]]: Supremum bounds of train=True parameters
        """
        A, b, G, h = self.get_polyhedron_hyperparams(theta, train_bool)
        if len(b) == 0:
            A = None
            b = None
        else:
            pass

        bounds = []
        n_var = G.shape[1]
        for i in range(n_var):
            bs = [-inf, inf]
            for j, coef in enumerate([1, -1]):
                c = np.zeros(n_var)
                c[i] = coef
                opt_res = linprog(c, A_ub=G, b_ub=h, A_eq=A, b_eq=b)
                if opt_res.success:
                    bs[j] = coef * opt_res.fun
            bounds.append(bs)

        return bounds

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

    def _raise_constraint_error(self, Ai, bi, operator):
        # Raises constraint error
        expr = self._pretty_expr(Ai, bi, operator)
        raise ValueError(f"Violates the equation: {expr}")

    def _safe_inf(self, vec):
        # Transform infinity to a number for 0*inf computations.
        vec = np.array(vec)
        vec[np.isinf(vec)] = NP_max
        return vec

    def __repr__(self):
        LessThan = self.all_constraints["LessThan"]
        EqualTo = self.all_constraints["EqualTo"]

        out = []
        for i in range(len(LessThan["A"])):
            eq_str = str(
                self._pretty_expr(LessThan["A"][i], LessThan["b"][i], "LessThan")
            )
            out.append(eq_str)

        for i in range(len(EqualTo["A"])):
            eq = self._pretty_expr(EqualTo["A"][i], EqualTo["b"][i], "EqualTo")
            lhs, rhs = str(eq.lhs), str(eq.rhs)
            eq_str = lhs + " == " + rhs
            out.append(eq_str)

        return "\n".join(out)


def linear_constraint_vector(linear_constraint_str: str, variables: list):
    """Takes a string expression and returns Ai, bi and operator.
    Args:
        linear_constraint_str (str): This is an string expression
        variables (list): list of variable names

    Returns:
        Tuple(list, list, list): Ai, bi, and operator, where
        Ai@x<=bi if operator=LessThan
        Ai@x=bi if operator=EqualTo

    Raises:
        ValueError: If equation is not in correct form for sympy to parse it
    """
    variable_order = dict(zip(variables, range(len(variables))))
    n_Theta = len(variables)

    if linear_constraint_str.find("==") != -1:
        # Equality constraint
        lhs, rhs = linear_constraint_str.split("==")
        lhs, rhs = sympy.sympify(lhs), sympy.sympify(rhs)
        expr = sympy.Eq(lhs, rhs).simplify()
        operator = "EqualTo"
    elif linear_constraint_str.find("<=") + linear_constraint_str.find(">=") > -2:
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

    lhs_params = [str(i) for i in lhs_params]
    rhs_params = [str(i) for i in rhs_params]

    if operator == "GreaterThan":
        lhs_coefs = [-i for i in lhs_coefs]
        rhs_constant = -rhs_constant
        operator = "LessThan"
    elif operator == "LessThan":
        rhs_coefs = [-i for i in rhs_coefs]
        lhs_constant = -lhs_constant
    else:
        rhs_coefs = [-i for i in rhs_coefs]

    Ai = n_Theta * [0]
    non_zero_params = lhs_params + rhs_params
    non_zero_coefs = lhs_coefs + rhs_coefs
    non_zero_indeces = [variable_order[i] for i in non_zero_params]
    for i, val in zip(non_zero_indeces, non_zero_coefs):
        Ai[i] = val

    bi = float(lhs_constant + rhs_constant)
    return Ai, bi, operator


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
        demes_params_desc = f"{param_name} of {dname}"
        if not b1:
            demes_params_desc += f" (epoch {j})"
    elif demes_event == "pulses":
        sources = " ".join(demo_dict[demes_event][i]["sources"])
        dest = demo_dict[demes_event][i]["dest"]
        demes_params_desc = f"{param_name} of the pulse from {sources} to {dest}"
    elif demes_event == "migrations":
        source = demo_dict[demes_event][i]["source"]
        dest = demo_dict[demes_event][i]["dest"]
        demes_params_desc = f"{param_name} of the migration from {source} to {dest}"
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

    return path, demes_params_desc


def reduce_linear_constraints(A, b):
    """To remove redundant inequalities from a system of linear inequalities.
    A @ x0 <= b
    A has shape (k, l) and b has (k,).
    This returns A, b with shape (r, l) and (r,) where r <= l

    Args:
        A (2d array): Description
        b (1d array): Description

    Returns:
        tuple(2d array, 1d array): Reduced A and b
    """
    i = 0
    while i < len(A):
        e = np.eye(len(A))[i]
        bi = b + e
        res = linprog(-A[i], A, bi, bounds=(None, None), method="simplex")
        if -res.fun <= b[i]:
            # constraint is redundant
            A = np.delete(A, i, axis=0)
            b = np.delete(b, i, axis=0)
        else:
            i += 1
    return A, b


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
        eq = eq.replace("<=", '≤')
        eq = eq.replace(">=", '≥')
        eq = eq.replace("==", '=')
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
    param_types = ["SizeParam", "RateParam", "ProportionParam", "TimeParam"]
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
                num = "{num:.3g}".format(num=cur.num)
                if params[key].train_it:
                    train = "&#9989"
                else:
                    train = "&#10060"
                table1_vals.append([name, num, train])
                for demes_params_desc in cur.demes_params_descs:
                    table2_vals.append([demes_params_desc, name])
        table1_vals = sorted(table1_vals, key=lambda x: x[0])
        body["table1"][param_type] = get_body(table1_vals, styles=styles["table1"])

        table2_vals = sorted(table2_vals, key=lambda x: x[1])
    body["table2"] = get_body(table2_vals, styles=styles["table2"])

    eq_table = []
    for eq in params._linear_constraints.user_constraint_dict.keys():
        eq_table.append([eq_to_html(eq)])
    eq_table = get_body(eq_table, styles=styles["table_eq"])

    # FIXME: Add the actual link to paper
    return f"""
<div style="display: inline-block; width: 30%;">
    <a href="https://github.com/jthlab/momi3" target="_blank">SOURCE CODE</a> <a href="https://thumbs.gfycat.com/LastBrilliantElk-mobile.mp4" target="_blank">PAPER</a>
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
