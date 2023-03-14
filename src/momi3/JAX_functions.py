from copy import deepcopy

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, hessian

from momi3.utils import sum_dicts, update
from momi3.Data import get_X_batches


@jax.jit
def multinomial_log_likelihood(P, Q, Q_sum):
    Q /= Q_sum
    Q = jnp.clip(1e-32, Q)
    return (P * jnp.log(Q)).sum()


class JAX_functions:
    def __init__(self, demo, T, jitted=False):
        self.demo = demo
        self._f = T.execute
        self.auxd = T._auxd
        self.leaves = T._leaves
        self._n_samples = T._num_samples

        def esfs_tensor_prod(theta_dict, X, auxd, demo, _f):
            demo_dict = demo.asdict()
            for paths, val in theta_dict.items():
                for path in paths:
                    update(demo_dict, path, val)
            return _f(demo_dict, X, auxd=auxd)

        if jitted:
            esfs_tensor_prod = jit(esfs_tensor_prod, static_argnames='_f')

        def esfs_vmap(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod):
            return jax.vmap(esfs_tensor_prod, (None, 0, None, None, None))(
                theta_dict, X_batch, auxd, demo, _f
            )

        if jitted:
            esfs_vmap = jit(esfs_vmap, static_argnames=('_f', 'esfs_tensor_prod'))

        def loglik_batch(
            theta_train_dict,
            theta_nuisance_dict,
            X_batch,
            sfs_batch,
            auxd,
            _f,
            esfs_tensor_prod,
            esfs_vmap,
            _n_samples=self._n_samples,
            leaves=self.leaves,
        ):
            theta_dict = deepcopy(theta_nuisance_dict)
            theta_dict.update(theta_train_dict)

            esfs_vec = esfs_vmap(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)

            Q = esfs_vec[3:]
            Q_sum = esfs_vec[0] - esfs_vec[1] - esfs_vec[2]
            P = sfs_batch

            return multinomial_log_likelihood(P, Q, Q_sum)

        if jitted:
            loglik_static_args = ('_f', 'esfs_tensor_prod', 'esfs_vmap')
            loglik_batch = jit(loglik_batch, static_argnames=loglik_static_args)

        loglik_and_grad_batch = value_and_grad(loglik_batch)
        if jitted:
            loglik_and_grad_batch = jit(loglik_and_grad_batch, static_argnames=loglik_static_args)

        hessian_batch = hessian(loglik_batch)
        if jitted:
            hessian_batch = jit(hessian_batch, static_argnames=loglik_static_args)

        self.esfs_tensor_prod = esfs_tensor_prod
        self.esfs_vmap = esfs_vmap
        self.loglik_batch = loglik_batch
        self.loglik_and_grad_batch = loglik_and_grad_batch
        self.hessian_batch = hessian_batch

    def esfs(self, theta_dict, num_deriveds, batch_size):
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

    def etbl(self, theta_dict):
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
        self, theta_train_dict, theta_nuisance_dict, data
    ):
        auxd = self.auxd
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
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        V = list(map(f, zip(X_batches, sfs_batches)))
        return sum(V)

    def loglik_and_grad(
        self, theta_train_dict, theta_nuisance_dict, data
    ):
        auxd = self.auxd
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
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        V, G = zip(*map(f, zip(X_batches, sfs_batches)))
        return sum(V), sum_dicts(G)

    def hessian(
        self, theta_train_dict, theta_nuisance_dict, data
    ):
        auxd = self.auxd
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
                _f,
                esfs_tensor_prod,
                esfs_vmap,
            )

        H = list(map(f, zip(X_batches, sfs_batches)))
        return sum_dicts(H)
