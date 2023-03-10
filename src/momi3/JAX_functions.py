# TODO: Add batch size to loglik calculation
from copy import deepcopy

import jax
import jax.numpy as jnp
from cached_property import cached_property

from momi3.utils import sum_dicts, update


@jax.jit
def multinomial_log_likelihood(P, Q, Q_sum):
    Q /= Q_sum
    Q = jnp.clip(1e-32, Q)
    return (P * jnp.log(Q)).sum()


class JAX_functions:
    def __init__(self, demo_dict, T, jitted=False, batch_size=None):
        self.demo_dict = demo_dict
        self._f = T.execute
        self.auxd = T._auxd
        self.leaves = T._leaves
        self._n_samples = T._num_samples
        self.jitted = jitted
        self.batch_size = batch_size

    def get_batches(self, data):
        batch_size = self.batch_size
        sfs = data.sfs
        num_deriveds = data.num_deriveds

        if batch_size is None:
            Num_deriveds = [num_deriveds]
            Sfs = [sfs]

        else:
            n_entries = len(next(iter(num_deriveds.values())))
            start = 0

            Num_deriveds = []
            Sfs = []
            for i in range(n_entries // batch_size):
                end = start + batch_size
                num_deriveds_batch = {
                    key: num_deriveds[key][start:end] for key in num_deriveds
                }
                sfs_batch = sfs[start:end]
                Num_deriveds.append(num_deriveds_batch)
                Sfs.append(sfs_batch)
                start = end

            if end != n_entries:
                pad_width = [0, n_entries - end]
                num_deriveds_batch = {
                    key: jnp.pad(num_deriveds[key][end:], pad_width)
                    for key in num_deriveds
                }
                sfs_batch = jnp.pad(sfs[end:], pad_width)
                Num_deriveds.append(num_deriveds_batch)
                Sfs.append(sfs_batch)

        return Num_deriveds, Sfs

    @cached_property
    def esfs_tensor_prod(self):
        demo_dict = self.demo_dict
        _f = self._f

        def _esfs_tensor_prod(theta_dict, X, auxd, demo_dict=demo_dict, _f=_f):
            demo_dict = deepcopy(demo_dict)
            for paths, val in theta_dict.items():
                for path in paths:
                    update(demo_dict, path, val)
            return _f(demo_dict, X, auxd=auxd)

        if self.jitted:
            _esfs_tensor_prod = jax.jit(_esfs_tensor_prod)

        return _esfs_tensor_prod

    @cached_property
    def esfs(self):
        demo_dict = self.demo_dict
        leaves = self.leaves
        _n_samples = self._n_samples
        esfs_tensor_prod = self.esfs_tensor_prod

        def _esfs(
            theta_dict,
            num_derived,
            auxd,
            demo_dict=demo_dict,
            leaves=leaves,
            _n_samples=_n_samples,
            esfs_tensor_prod=esfs_tensor_prod,
        ):
            X = {}
            for pop in leaves:
                # some ghost populations may not be sampled. then they have trivial partial leaf likelihood.
                ns = _n_samples.get(pop, 0)
                d = num_derived.get(pop, 0)
                X[pop] = jax.nn.one_hot(jnp.array([d]), ns + 1)[0]
            return esfs_tensor_prod(theta_dict, X, auxd)

        return _esfs

    @cached_property
    def esfs_vmap(self):
        esfs_tensor_prod = self.esfs_tensor_prod

        def _esfs_vmap(theta_dict, X_batch, auxd, esfs_tensor_prod=esfs_tensor_prod):
            return jax.vmap(esfs_tensor_prod, (None, 0, None))(
                theta_dict, X_batch, auxd
            )

        if self.jitted:
            _esfs_vmap = jax.jit(_esfs_vmap)
        return _esfs_vmap

    @cached_property
    def etbl(self):
        demo_dict = self.demo_dict
        leaves = self.leaves
        _n_samples = self._n_samples
        esfs_tensor_prod = self.esfs_tensor_prod

        def _etbl(
            theta_dict,
            auxd,
            demo_dict=demo_dict,
            leaves=leaves,
            _n_samples=_n_samples,
            esfs_tensor_prod=esfs_tensor_prod,
        ):
            vecs = [{}, {}, {}]
            for pop in leaves:
                ns = _n_samples.get(pop, 0)
                vecs[0][pop] = jnp.ones(ns + 1, dtype="f")
                vecs[1][pop] = jax.nn.one_hot(jnp.array([0]), ns + 1)[0]
                vecs[2][pop] = jax.nn.one_hot(jnp.array([ns]), ns + 1)[0]

            ret0 = esfs_tensor_prod(theta_dict, vecs[0], auxd)
            ret1 = esfs_tensor_prod(theta_dict, vecs[1], auxd)
            ret2 = esfs_tensor_prod(theta_dict, vecs[2], auxd)
            return (ret0 - ret1 - ret2).flatten()[0]

        return _etbl

    @cached_property
    def loglik_batch(self):
        esfs_vmap = self.esfs_vmap
        _n_samples = self._n_samples

        def _loglik_batch(
            theta_train_dict,
            theta_nuisance_dict,
            X_batch,
            sfs_batch,
            auxd,
            _n_samples=_n_samples,
            esfs_vmap=esfs_vmap,
        ):
            theta_dict = deepcopy(theta_nuisance_dict)
            theta_dict.update(theta_train_dict)

            esfs_vec = esfs_vmap(theta_dict, X_batch, auxd)

            Q = esfs_vec[3:]
            Q_sum = esfs_vec[0] - esfs_vec[1] - esfs_vec[2]
            P = sfs_batch

            return multinomial_log_likelihood(P, Q, Q_sum)

        if self.jitted:
            _loglik_batch = jax.jit(_loglik_batch)
        return _loglik_batch

    @cached_property
    def loglik(self):
        loglik_batch = self.loglik_batch

        def _loglik(
            theta_train_dict, theta_nuisance_dict, data, auxd, loglik_batch=loglik_batch
        ):
            X_batches, sfs_batches = data.X_batches, data.sfs_batches

            def f(x):
                return loglik_batch(
                    theta_train_dict, theta_nuisance_dict, x[0], x[1], auxd
                )

            V = list(map(f, zip(X_batches, sfs_batches)))
            return sum(V)

        return _loglik

    @cached_property
    def loglik_and_grad(self):
        loglik_batch = self.loglik_batch
        _loglik_and_grad_batch = jax.value_and_grad(loglik_batch)
        if self.jitted:
            _loglik_and_grad_batch = jax.jit(_loglik_and_grad_batch)

        def _loglik_and_grad(
            theta_train_dict,
            theta_nuisance_dict,
            data,
            auxd,
            _loglik_and_grad_batch=_loglik_and_grad_batch,
        ):
            X_batches, sfs_batches = data.X_batches, data.sfs_batches

            def f(x):
                return _loglik_and_grad_batch(
                    theta_train_dict, theta_nuisance_dict, x[0], x[1], auxd
                )

            V, G = zip(*map(f, zip(X_batches, sfs_batches)))
            return sum(V), sum_dicts(G)

        return _loglik_and_grad

    @cached_property
    def hessian(self):
        loglik_batch = self.loglik_batch
        _hessian_batch = jax.hessian(loglik_batch)
        if self.jitted:
            _hessian_batch = jax.jit(_hessian_batch)

        def _hessian(
            theta_train_dict,
            theta_nuisance_dict,
            data,
            auxd,
            _hessian_batch=_hessian_batch,
        ):
            X_batches, sfs_batches = data.X_batches, data.sfs_batches

            def f(x):
                return _hessian_batch(
                    theta_train_dict, theta_nuisance_dict, x[0], x[1], auxd
                )

            H = list(map(f, zip(X_batches, sfs_batches)))
            return sum_dicts(H)

        return _hessian
