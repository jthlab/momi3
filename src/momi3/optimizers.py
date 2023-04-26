from collections import namedtuple
from time import time
from typing import Callable, Union

import cvxpy
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxopt import ProjectedGradient
from sparse import COO
from tqdm.autonotebook import tqdm, trange

import optax

from momi3.Params import Params


def is_theta_valid(theta, A, b, G, h, atol=1e-8, rtol=1e-5):
    b1 = jnp.allclose(A @ theta, b, atol=atol, rtol=rtol)
    b2 = jnp.all(G @ theta <= h)
    return b1 & b2


def cvxpy_projection_polyhedron(theta, hyperparams_proj):
    A, b, G, h = hyperparams_proj
    x = cvxpy.Variable(len(theta))
    prob = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.sum_squares(theta - x)), [G @ x <= h, A @ x == b]
    )
    prob.solve()
    return x.value


def ProjectedGradient_optimizer(
    negative_loglik_with_gradient: Callable,
    params: Params,
    jsfs: Union[COO, jnp.ndarray, np.ndarray],
    stepsize: float,
    maxiter: int,
    sampled_demes: tuple[str],
    theta_train_dict_0: dict[str, float] = None,
    htol: float = 0.0,
    monitor_training: bool = False,
) -> namedtuple:
    stepsize = stepsize / jsfs.sum()  # scale the step size

    theta_train_keys = params._train_keys

    if theta_train_dict_0 is None:
        theta_train_0 = jnp.array(params._theta_train)
    else:
        theta_train_0 = jnp.array([theta_train_dict_0[i] for i in theta_train_keys])

    A, b, G, h = params._polyhedron_hyperparams(htol)

    def FwG(theta_train, theta_train_keys=theta_train_keys):
        theta_train_dict = {}
        for paths, value in zip(theta_train_keys, theta_train):
            theta_train_dict[paths] = float(value)
        return negative_loglik_with_gradient(
            params, jsfs, theta_train_dict, return_array=True
        )

    pg = ProjectedGradient(
        fun=FwG,
        projection=cvxpy_projection_polyhedron,
        jit=False,
        stepsize=stepsize,
        value_and_grad=True,
        maxiter=maxiter,
    )

    ret_dict = {
        "loglik_0": 0,
        "loglik_n": 0,
        "loglik_grad": 0,
        "theta_hat": 0,
        "pg_state": 0,
    }

    show_boundry_message = True
    if monitor_training:
        st = time()
        obj0, grad0 = FwG(theta_train_0)
        elapsed = time() - st
        print(f"First run took {elapsed:.2f} seconds")
        pg_state = pg.init_state(theta_train_0)
        loss = []
        theta_train_hat = theta_train_0.copy()
        for i in trange(maxiter):
            theta_train_hat, pg_state = pg.update(
                theta_train_hat, pg_state, hyperparams_proj=(A, b, G, h)
            )
            neg_log_lik = FwG(theta_train_hat)[0]

            if show_boundry_message:
                if jnp.any(jnp.isclose(G @ theta_train_hat, h)):
                    # logging.warn("Try running it with smaller a step_size")
                    tqdm.write(
                        "theta + step * gradient has hit a boundry. Try running it with a smaller step_size"
                    )

                    show_boundry_message = False

            # Messages:
            tqdm.write(f"loglik={-neg_log_lik:.5g}")
            params_message = ", ".join(
                [f"{i}={j:.2g}" for i, j in zip(theta_train_keys, theta_train_hat)]
            )
            tqdm.write(f"Values: {params_message}")
            tqdm.write(10 * "=")

            loss.append(pg_state.error)

        plt.plot(range(1, maxiter + 1), loss)
        plt.ylabel("Error")  # TODO: Not sure what this error is
        plt.xlabel("Iteration Number")

    else:
        opt_result = pg.run(theta_train_0, hyperparams_proj=(A, b, G, h))
        theta_train_hat = opt_result.params
        pg_state = opt_result.state

    obj0, grad0 = FwG(theta_train_0)
    obj, grad = FwG(theta_train_hat)

    ret_dict["loglik_0"] = -obj0
    ret_dict["loglik_n"] = -obj
    ret_dict["likelihood_ratio"] = -2 * (ret_dict["loglik_0"] - ret_dict["loglik_n"])
    ret_dict["loglik_grad"] = -grad
    ret_dict["theta_train_hat"] = dict(
        zip(theta_train_keys, [float(i) for i in theta_train_hat])
    )
    ret_dict["pg_state"] = pg_state
    ProjectedGradientReturn = namedtuple("ProjectedGradientReturn", ret_dict.keys())
    ret_tuple = ProjectedGradientReturn._make(ret_dict.values())
    return ret_tuple


def optax_step(optimizer, f, theta_train_dict, opt_state):
    loss_value, grads = f(theta_train_dict)
    updates, opt_state = optimizer.update(grads, opt_state, theta_train_dict)
    theta_train_dict = optax.apply_updates(theta_train_dict, updates)
    return theta_train_dict, opt_state, loss_value


def optax_for_momi(
    optimizer, momi, params, jsfs, niter, transformed=True, theta_train_dict=None, opt_state=None, history=None
):

    if theta_train_dict is None:
        theta_train_dict = params.theta_train_dict(transformed)

    if history is None:
        history = dict(LLs=[], ttds=[])

    train_keys = list(theta_train_dict)
    theta_train = jnp.array(list(theta_train_dict.values()))

    if opt_state is None:
        opt_state = optimizer.init(theta_train)

    def f(theta_train, train_keys=train_keys):
        v, g = momi.negative_loglik_with_gradient(
            params, jsfs, dict(zip(train_keys, theta_train)), transformed=transformed
        )
        g = jnp.array([g[i] for i in train_keys])
        return v, g

    for i in trange(niter):
        history['ttds'].append(dict(zip(train_keys, theta_train)))
        theta_train, opt_state, loss_value = optax_step(optimizer, f, theta_train, opt_state)
        history['LLs'].append(loss_value)

    return dict(zip(train_keys, theta_train)), opt_state, history
