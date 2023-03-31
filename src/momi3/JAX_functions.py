import logging

import jax
import jax.numpy as jnp
from jax import checkpoint, hessian, jit, value_and_grad

from momi3.Data import Data, get_X_batches
from momi3.utils import update

logging.basicConfig(level=logging.WARNING)


def multinomial_log_likelihood(P, Q, Q_sum):
    Q /= Q_sum
    Q = jnp.clip(1e-32, Q)
    return (P * jnp.log(Q)).sum()


def esfs_tensor_prod(theta_dict, X, auxd, demo, _f):
    demo_dict = demo.asdict()
    for paths, val in theta_dict.items():
        for path in paths:
            update(demo_dict, path, val)
    return _f(demo_dict, X, auxd)


def esfs_map(theta_dict, X, auxd, demo, _f, esfs_tensor_prod, low_memory=False):
    # X[pop].shape = (A, B, C)
    # A: jax.lax.map size
    # B: jax.vmap size
    # C: sample size + 1
    def f(X_batch):
        return jax.vmap(esfs_tensor_prod, (None, 0, None, None, None))(
            theta_dict, X_batch, auxd, demo, _f
        )

    if low_memory:
        f = checkpoint(f)
    return jax.vmap(f)(X).flatten()
    return jax.lax.map(f, X).flatten()


def esfs_mapX(theta_dict, X, auxd, demo, _f, esfs_tensor_prod):
    # This is an experimental fun for including pmap in calc map(pmap(vmap))(esfs)
    # X[pop].shape = (A, B, C, D)
    # A: jax.lax.map size
    # B: jax.pmap size
    # C: jax.vmap size
    # D: sample size + 1
    def f(X_batch):
        return jax.pmap(
            jax.vmap(esfs_tensor_prod, (None, 0, None, None, None)),
            in_axes=(None, 0, None, None, None),
            static_broadcasted_argnums=(4),
        )(theta_dict, X_batch, auxd, demo, _f)

    return jax.lax.map(f, X).flatten()


def loglik_batch(
    theta_train_dict,
    theta_nuisance_dict,
    X_batch,
    sfs_batch,
    auxd,
    demo,
    _f,
    esfs_tensor_prod,
    esfs_map,
):
    theta_dict = theta_train_dict | theta_nuisance_dict

    esfs_vec = esfs_map(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)

    Q = esfs_vec[3:]
    Q_sum = esfs_vec[0] - esfs_vec[1] - esfs_vec[2]
    P = sfs_batch

    return multinomial_log_likelihood(P, Q, Q_sum)


loglik_and_grad_batch = value_and_grad(loglik_batch)
hessian_batch = hessian(loglik_batch)


def loglik_batch_transformed(
    transformed_theta_train_dict,
    transformed_theta_nuisance_dict,
    X_batch,
    sfs_batch,
    auxd,
    demo,
    _f,
    esfs_tensor_prod,
    esfs_map,
):
    theta_dict = []
    for i, j in zip(transformed_theta_train_dict, transformed_theta_nuisance_dict):
        theta_dict.append(i | j)

    # eta
    for key in theta_dict[0]:
        val = theta_dict[0][key]
        theta_dict[0][key] = jnp.exp(val)

    # rho and pi
    for i in [1, 2]:
        for key in theta_dict[i]:
            val = theta_dict[i][key]
            theta_dict[i][key] = jax.nn.sigmoid(val)

    # tau
    diff_tau_dict = theta_dict[3]
    the_next = tuple(diff_tau_dict[(("init",),)])[0]
    cum_val = diff_tau_dict[(("init",),)][the_next]
    tau_dict = {the_next: jnp.array(cum_val, dtype="f")}
    for i in range(len(diff_tau_dict) - 1):
        cur_dict = diff_tau_dict[the_next]
        the_next = tuple(cur_dict)[0]
        cum_val += jnp.exp(cur_dict[the_next])
        tau_dict[the_next] = cum_val
    theta_dict[3] = tau_dict

    theta_dict = theta_dict[0] | theta_dict[1] | theta_dict[2] | theta_dict[3]

    esfs_vec = esfs_map(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)

    Q = esfs_vec[3:]
    Q_sum = esfs_vec[0] - esfs_vec[1] - esfs_vec[2]
    P = sfs_batch

    return multinomial_log_likelihood(P, Q, Q_sum)


loglik_and_grad_batch_transformed = value_and_grad(loglik_batch_transformed)
hessian_batch_transformed = hessian(loglik_batch_transformed)


class JAX_functions:
    def __init__(
        self,
        demo,
        T,
        low_memory=False,
        jitted=False,
        esfs_tensor_prod=esfs_tensor_prod,
        esfs_map=esfs_map,
        loglik_batch=loglik_batch,
        loglik_and_grad_batch=loglik_and_grad_batch,
        hessian_batch=hessian_batch,
        esfs_mapX=esfs_mapX,
    ):
        self.demo = demo
        self._f = T.execute
        self.auxd = T._auxd
        self.leaves = tuple(T._leaves)
        self._n_samples = T._num_samples
        self.n_devices = jax.device_count()
        self.esfs_map = (
            lambda theta_dict, X, auxd, demo, _f, esfs_tensor_prod: esfs_map(
                theta_dict, X, auxd, demo, _f, esfs_tensor_prod, low_memory=low_memory
            )
        )

        if jitted:
            esfs_tensor_prod = jit(esfs_tensor_prod, static_argnames="_f")
            self.esfs_map = jit(
                self.esfs_map, static_argnames=("_f", "esfs_tensor_prod")
            )
            esfs_mapX = jax.jit(esfs_mapX, static_argnums=(4, 5))

            loglik_static_nums = (6, 7, 8)
            loglik_batch = jit(loglik_batch, static_argnums=loglik_static_nums)
            loglik_and_grad_batch = jit(
                loglik_and_grad_batch, static_argnums=loglik_static_nums
            )
            hessian_batch = jit(hessian_batch, static_argnums=loglik_static_nums)

        self.esfs_tensor_prod = esfs_tensor_prod
        self.esfs_map = esfs_map
        self.loglik_batch = loglik_batch
        self.loglik_and_grad_batch = loglik_and_grad_batch
        self.hessian_batch = hessian_batch
        self.esfs_mapX = esfs_mapX

    def esfs(
        self,
        theta_dict: dict[tuple, float],
        num_deriveds: dict[str, jnp.ndarray],
        batch_size: int,
    ) -> jnp.ndarray:
        """Calculate expected site frequency spectrum for the sample config in num_deriveds.
            Example for The Gutenkunst et al (2009) out-of-Africa.
            Below example calculates 4 sfs configation -(0, 1, 0), (1, 0, 0), (0, 2, 1), (1, 0, 0)- by setting values
            for 2 different path tuples.
                esfs(
                    theta_dict = {
                        (('demes', 0, 'epochs', 0, 'start_size'), ('demes', 0, 'epochs', 0, 'end_size')): 7300.0,
                        (('demes', 1, 'epochs', 0, 'end_size'), ('demes', 1, 'epochs', 0, 'start_size')): 12300.0
                    },
                    num_deriveds = {
                        'YRI': jnp.array([0, 1, 0, 1]),
                        'CEU': jnp.array([1, 0, 2, 0]),
                        'CHB': jnp.array([0, 0, 1, 0])
                    },
                    batch_size = 10
                )

        Args:
            theta_dict (dict[str, float]): Values of the parameters. Keys are the tuple of paths.
            num_deriveds (dict[str, int]): Number of deriveds for each sampled population.
            batch_size (int): Batch size for jax.vmap. This variable controls the memory usage. Users should
                lower the value if they are getting and OOM error.

        Returns:
            jnp.ndarray: Expected SFS vector for the configurations in num_deriveds.
        """
        leaves = self.leaves
        _n_samples = self._n_samples
        sampled_demes = tuple(_n_samples)
        sample_sizes = tuple(_n_samples[pop] for pop in sampled_demes)
        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_map = self.esfs_map

        deriveds = tuple(
            tuple(int(i) for i in num_deriveds[pop]) for pop in sampled_demes
        )
        n_entries = len(deriveds[0])

        X = get_X_batches(
            sampled_demes,
            sample_sizes,
            leaves,
            deriveds,
            batch_size,
            add_etbl_vecs=False,
        )

        n_devices = self.n_devices

        if n_devices == 1:
            ret = esfs_map(
                theta_dict,
                {pop: X[pop][0] for pop in X},
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
            )
        else:
            pmap_fun = jax.pmap(
                esfs_map,
                in_axes=(None, 0, None, None, None),
                static_broadcasted_argnums=(4, 5),
            )

            ret = pmap_fun(theta_dict, X, auxd, demo, _f, esfs_tensor_prod)

        return ret.flatten()[:n_entries]

    def etbl(self, theta_dict: dict[tuple, float]) -> float:
        """Calculate the total branch length for the given values.
            Example for The Gutenkunst et al (2009) out-of-Africa.
            Below example calculates etbl by setting values for 2 different path tuples.
                esfs(
                    theta_dict = {
                        (('demes', 0, 'epochs', 0, 'start_size'), ('demes', 0, 'epochs', 0, 'end_size')): 7300.0,
                        (('demes', 1, 'epochs', 0, 'end_size'), ('demes', 1, 'epochs', 0, 'start_size')): 12300.0
                    }
                )

        Args:
            theta_dict (dict[str, float]): Values of the parameters. Keys are the tuple of paths.

        Returns:
            float: Expected Total Branch Length for the configurations in num_deriveds.
        """
        demo = self.demo
        leaves = self.leaves
        _n_samples = self._n_samples
        _f = self._f
        auxd = self.auxd
        esfs_tensor_prod = self.esfs_tensor_prod

        X_batch = [{}, {}, {}]
        for pop in leaves:
            ns = _n_samples.get(pop, 0)
            X_batch[0][pop] = jnp.ones(ns + 1, dtype="f")
            X_batch[1][pop] = jax.nn.one_hot(jnp.array([0]), ns + 1)[0]
            X_batch[2][pop] = jax.nn.one_hot(jnp.array([ns]), ns + 1)[0]

        ret0 = esfs_tensor_prod(theta_dict, X_batch[0], auxd, demo, _f)
        ret1 = esfs_tensor_prod(theta_dict, X_batch[1], auxd, demo, _f)
        ret2 = esfs_tensor_prod(theta_dict, X_batch[2], auxd, demo, _f)
        return (ret0 - ret1 - ret2).flatten()[0]

    def _run_fun(
        self,
        fun,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data,
    ):
        # This fun is called by
        # loglik: f(theta, data)
        # loglik_and_grad: f(theta, data), df(theta, data)
        # hessian d^2 f(theta, data)

        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_map = self.esfs_map

        X_batches, sfs_batches = data.X_batches, data.sfs_batches

        n_devices = self.n_devices

        logging.warn("X: " + " ".join([str(X_batches[pop].shape) for pop in X_batches]))
        logging.warn("sfs: " + str(sfs_batches.shape))

        if n_devices == 1:
            return fun(
                theta_train_dict,
                theta_nuisance_dict,
                {pop: X_batches[pop][0] for pop in X_batches},
                sfs_batches[0],
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_map,
            )
        else:
            # return fun(
            #     theta_train_dict,
            #     theta_nuisance_dict,
            #     X_batches,
            #     sfs_batches,
            #     auxd,
            #     demo,
            #     _f,
            #     esfs_tensor_prod,
            #     self.esfs_mapX
            # )

            pmap_fun = jax.pmap(
                fun,
                in_axes=(None, None, 0, 0, None, None, None, None, None),
                static_broadcasted_argnums=(6, 7, 8),
            )

            return pmap_fun(
                theta_train_dict,
                theta_nuisance_dict,
                X_batches,
                sfs_batches,
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_map,
            )

    def loglik(
        self,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data,
    ) -> float:
        """Returns multinomial log-likelihood for the given data and theta.
            Example for The Gutenkunst et al (2009) out-of-Africa.
            Below example calculates loglik by setting values for 2 different path tuples for training theta
            and 4 for non-training. For likelihood it doesn't matter if a path is in train_dict or in nuisance_dict.
            This will matter for gradient for likelihood. And mydata, is a loaded jsfs. See Momi._get_data
                loglik(
                    theta_train_dict = {
                        (('demes', 0, 'epochs', 0, 'start_size'), ('demes', 0, 'epochs', 0, 'end_size')): 7300.0,
                        (('demes', 1, 'epochs', 0, 'end_size'), ('demes', 1, 'epochs', 0, 'start_size')): 12300.0
                    },
                    theta_nuisance_dict = {
                        (('demes', 4, 'epochs', 0, 'start_size'),): 1000.0,
                        (('demes', 4, 'epochs', 0, 'end_size'),): 29725.0,
                        (('demes', 5, 'epochs', 0, 'start_size'),): 510.0,
                        (('demes', 5, 'epochs', 0, 'end_size'),): 54090.0,
                        (('migrations', 0, 'rate'), ('migrations', 1, 'rate')): 0.00025
                    },
                    data = mydata
                )
        Args:
            theta_train_dict (dict[tuple, float]): Training values of the parameters. Keys are the tuple of paths.
            theta_nuisance_dict (dict[tuple, float]): Values of the parameters. Keys are the tuple of paths.
            data (Data): Data object. It stores, sfs and leaf likelihoods.
        """

        n_devices = self.n_devices

        V = self._run_fun(
            self.loglik_batch, theta_train_dict, theta_nuisance_dict, data
        )

        if n_devices > 1:
            V = V.sum()

        return V

    def loglik_and_grad(
        self,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data,
        batch=True,
    ) -> tuple[float, dict[tuple, float]]:
        """Returns multinomial log-likelihood and gradient of theta_train_dict for the given data and theta.
            Example for The Gutenkunst et al (2009) out-of-Africa.
            Below example calculates loglik by setting values for 2 different path tuples for training theta
            and 4 for non-training. And mydata, is a loaded jsfs. See Momi._get_data
                loglik(
                    theta_train_dict = {
                        (('demes', 0, 'epochs', 0, 'start_size'), ('demes', 0, 'epochs', 0, 'end_size')): 7300.0,
                        (('demes', 1, 'epochs', 0, 'end_size'), ('demes', 1, 'epochs', 0, 'start_size')): 12300.0
                    },
                    theta_nuisance_dict = {
                        (('demes', 4, 'epochs', 0, 'start_size'),): 1000.0,
                        (('demes', 4, 'epochs', 0, 'end_size'),): 29725.0,
                        (('demes', 5, 'epochs', 0, 'start_size'),): 510.0,
                        (('demes', 5, 'epochs', 0, 'end_size'),): 54090.0,
                        (('migrations', 0, 'rate'), ('migrations', 1, 'rate')): 0.00025
                    },
                    data = mydata
                )
        Args:
            theta_train_dict (dict[tuple, float]): Training values of the parameters. Keys are the tuple of paths.
            theta_nuisance_dict (dict[tuple, float]): Values of the parameters. Keys are the tuple of paths.
            data (Data): Data object. It stores, sfs and leaf likelihoods.
        """

        n_devices = self.n_devices

        V, G = self._run_fun(
            self.loglik_and_grad_batch, theta_train_dict, theta_nuisance_dict, data
        )

        if n_devices > 1:
            V = V.sum()
            G = {i: G[i].sum() for i in G}

        return V, G

    def hessian(
        self,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data,
    ) -> tuple[float, dict[tuple[tuple, tuple], float]]:
        """Returns hessian of theta_train_dict for the given data and theta.
            Example for The Gutenkunst et al (2009) out-of-Africa.
            Below example calculates loglik by setting values for 2 different path tuples for training theta
            and 4 for non-training. And mydata, is a loaded jsfs. See Momi._get_data
                loglik(
                    theta_train_dict = {
                        (('demes', 0, 'epochs', 0, 'start_size'), ('demes', 0, 'epochs', 0, 'end_size')): 7300.0,
                        (('demes', 1, 'epochs', 0, 'end_size'), ('demes', 1, 'epochs', 0, 'start_size')): 12300.0
                    },
                    theta_nuisance_dict = {
                        (('demes', 4, 'epochs', 0, 'start_size'),): 1000.0,
                        (('demes', 4, 'epochs', 0, 'end_size'),): 29725.0,
                        (('demes', 5, 'epochs', 0, 'start_size'),): 510.0,
                        (('demes', 5, 'epochs', 0, 'end_size'),): 54090.0,
                        (('migrations', 0, 'rate'), ('migrations', 1, 'rate')): 0.00025
                    },
                    data = mydata
                )
        Args:
            theta_train_dict (dict[tuple, float]): Training values of the parameters. Keys are the tuple of paths.
            theta_nuisance_dict (dict[tuple, float]): Values of the parameters. Keys are the tuple of paths.
            data (Data): Data object. It stores, sfs and leaf likelihoods.
        """

        n_devices = self.n_devices

        H = self._run_fun(
            self.hessian_batch, theta_train_dict, theta_nuisance_dict, data
        )

        if n_devices > 1:
            H = {i: {j: H[i][j].sum() for j in H[i]} for i in H}

        return H
