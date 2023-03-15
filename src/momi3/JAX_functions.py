from copy import deepcopy

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, hessian

from momi3.utils import sum_dicts, update
from momi3.Data import get_X_batches, Data


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


def esfs_vmap(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod):
    return jax.vmap(esfs_tensor_prod, (None, 0, None, None, None))(
        theta_dict, X_batch, auxd, demo, _f
    )


def loglik_batch(
    theta_train_dict,
    theta_nuisance_dict,
    X_batch,
    sfs_batch,
    auxd,
    demo,
    _f,
    esfs_tensor_prod,
    esfs_vmap,
):
    theta_dict = deepcopy(theta_nuisance_dict)
    theta_dict.update(theta_train_dict)

    esfs_vec = esfs_vmap(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)

    Q = esfs_vec[3:]
    Q_sum = esfs_vec[0] - esfs_vec[1] - esfs_vec[2]
    P = sfs_batch

    return multinomial_log_likelihood(P, Q, Q_sum)


loglik_and_grad_batch = value_and_grad(loglik_batch)


hessian_batch = hessian(loglik_batch)


class JAX_functions:
    def __init__(
        self,
        demo,
        T,
        jitted=False,
        esfs_tensor_prod=esfs_tensor_prod,
        esfs_vmap=esfs_vmap,
        loglik_batch=loglik_batch,
        loglik_and_grad_batch=loglik_and_grad_batch,
        hessian_batch=hessian_batch
    ):
        self.demo = demo
        self._f = T.execute
        self.auxd = T._auxd
        self.leaves = T._leaves
        self._n_samples = T._num_samples

        if jitted:
            esfs_tensor_prod = jit(esfs_tensor_prod, static_argnames='_f')
            esfs_vmap = jit(esfs_vmap, static_argnames=('_f', 'esfs_tensor_prod'))

            loglik_static_args = ('_f', 'esfs_tensor_prod', 'esfs_vmap')
            loglik_batch = jit(loglik_batch, static_argnames=loglik_static_args)
            loglik_and_grad_batch = jit(loglik_and_grad_batch, static_argnames=loglik_static_args)
            hessian_batch = jit(hessian_batch, static_argnames=loglik_static_args)

        self.esfs_tensor_prod = esfs_tensor_prod
        self.esfs_vmap = esfs_vmap
        self.loglik_batch = loglik_batch
        self.loglik_and_grad_batch = loglik_and_grad_batch
        self.hessian_batch = hessian_batch

    def esfs(self, theta_dict: dict[tuple, float], num_deriveds: dict[str, jnp.ndarray], batch_size: int) -> jnp.ndarray:
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
        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_vmap = self.esfs_vmap

        X_batches = get_X_batches(num_deriveds, leaves, _n_samples, batch_size, add_etbl_vecs=False)
        f = lambda X_batch: esfs_vmap(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)
        V = list(map(f, X_batches))
        return jnp.concatenate(V)

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

    def loglik(
        self, theta_train_dict: dict[tuple, float], theta_nuisance_dict: dict[tuple, float], data: Data
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
        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_vmap = self.esfs_vmap
        loglik_batch = self.loglik_batch

        X_batches, sfs_batches = data.X_batches, data.sfs_batches

        def f(x):
            X_batch, sfs_batch = x
            return loglik_batch(
                theta_train_dict,
                theta_nuisance_dict,
                X_batch,
                sfs_batch,
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        V = list(map(f, zip(X_batches, sfs_batches)))
        return sum(V)

    def loglik_and_grad(
        self, theta_train_dict: dict[tuple, float], theta_nuisance_dict: dict[tuple, float], data: Data
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
        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_vmap = self.esfs_vmap
        loglik_and_grad_batch = self.loglik_and_grad_batch

        X_batches, sfs_batches = data.X_batches, data.sfs_batches

        def f(x):
            X_batch, sfs_batch = x
            return loglik_and_grad_batch(
                theta_train_dict,
                theta_nuisance_dict,
                X_batch,
                sfs_batch,
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        V, G = zip(*map(f, zip(X_batches, sfs_batches)))
        return sum(V), sum_dicts(G)

    def hessian(
        self, theta_train_dict: dict[tuple, float], theta_nuisance_dict: dict[tuple, float], data: Data
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
        auxd = self.auxd
        demo = self.demo
        _f = self._f
        esfs_tensor_prod = self.esfs_tensor_prod
        esfs_vmap = self.esfs_vmap
        hessian_batch = self.hessian_batch

        X_batches, sfs_batches = data.X_batches, data.sfs_batches

        def f(x):
            X_batch, sfs_batch = x
            return hessian_batch(
                theta_train_dict,
                theta_nuisance_dict,
                X_batch,
                sfs_batch,
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        H = list(map(f, zip(X_batches, sfs_batches)))
        return sum_dicts(H)
