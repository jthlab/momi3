from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import tqdm

from momi3.Data import Data
from momi3.JAX_functions import esfs_tensor_prod, esfs_map
from momi3.JAX_functions import loglik_batch_transformed as loglik_batch
from momi3.JAX_functions import loglik_and_grad_batch_transformed as loglik_and_grad_batch
from momi3.JAX_functions import hessian_batch_transformed as hessian_batch

import jax
from jax import jit

import math
import numdifftools as nd
from copy import deepcopy
import logging


class NumHessian:
    def __init__(self, momi: Momi, params: Params):
        self.momi = momi
        self.params = params
        self.theta_train_dict = params._theta_train_dict
        self.theta_nuisance_dict = params._theta_nuisance_dict
        self.JF = momi._JAX_functions
        self.auxd = momi._auxd
        self.demo = params.demo_graph

    def loglik_for_numdiff(
        self,
        val,
        paths,
        X_batch,
        sfs_batch
    ):

    	theta_train_dict = deepcopy(self.theta_train_dict)
    	theta_train_dict[paths] = float(val)
    	return self.JF.loglik_batch(
            theta_train_dict,
            self.theta_nuisance_dict,
            X_batch,
            sfs_batch,
            self.auxd,
            self.demo,
            self.JF._f,
            self.JF.esfs_tensor_prod,
            self.JF.esfs_map,
        )

    def numerical_diag_of_hessian(self, X_batch, sfs_batch, num_steps=10):
	    train_keys = self.params._train_keys

	    num_hess = {}
	    for key in tqdm(train_keys):
	        paths = self.params._params_to_paths[key]
	        val = self.theta_train_dict[paths]
	        df = nd.Derivative(
	            self.loglik_for_numdiff, step=None, n=2, num_steps=num_steps, method='central'
	        )
	        num_hess[key] = float(df(val, paths, X_batch, sfs_batch))

	    return num_hess

    def numerical_FIM_uncert(self, X_batch, sfs_batch, num_steps=10):
	    diag_FIM = self.numerical_diag_of_hessian(X_batch, sfs_batch, num_steps=num_steps)
	    return {key: math.sqrt(1 / abs(diag_FIM[key])) for key in diag_FIM}


class TransformedJAX_functions:
    def __init__(
        self,
        demo,
        T,
        jitted=False,
        esfs_tensor_prod=esfs_tensor_prod,
        esfs_map=esfs_map,
        loglik_batch=loglik_batch,
        loglik_and_grad_batch=loglik_and_grad_batch,
        hessian_batch=hessian_batch
    ):

        self.demo = demo
        self._f = T.execute
        self.auxd = T._auxd
        self.leaves = tuple(T._leaves)
        self._n_samples = T._num_samples
        self.n_devices = jax.device_count()

        if jitted:
            esfs_tensor_prod = jit(esfs_tensor_prod, static_argnames='_f')
            esfs_map = jit(esfs_map, static_argnames=('_f', 'esfs_tensor_prod'))

            loglik_static_nums = (6, 7, 8)
            loglik_batch = jit(loglik_batch, static_argnums=loglik_static_nums)
            loglik_and_grad_batch = jit(loglik_and_grad_batch, static_argnums=loglik_static_nums)
            hessian_batch = jit(hessian_batch, static_argnums=loglik_static_nums)

        self.esfs_tensor_prod = esfs_tensor_prod
        self.esfs_map = esfs_map
        self.loglik_batch = loglik_batch
        self.loglik_and_grad_batch = loglik_and_grad_batch
        self.hessian_batch = hessian_batch

    def _run_fun(
        self,
        fun,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data
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

        logging.warning(' '.join([str(X_batches[pop].shape) for pop in X_batches]))

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
                esfs_map
            )
        else:

            pmap_fun = jax.pmap(
                fun,
                in_axes=(None, None, 0, 0, None, None, None, None, None),
                static_broadcasted_argnums=(6, 7, 8)
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
                esfs_map
            )

    def loglik(
        self,
        theta_train_dict: dict[tuple, float],
        theta_nuisance_dict: dict[tuple, float],
        data: Data
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
            self.loglik_batch,
            theta_train_dict,
            theta_nuisance_dict,
            data
        )

        if n_devices > 1:
            V = V.sum()

        return V

    def loglik_and_grad(
        self, theta_train_dict: dict[tuple, float], theta_nuisance_dict: dict[tuple, float], data: Data, batch=True
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
            self.loglik_and_grad_batch,
            theta_train_dict,
            theta_nuisance_dict,
            data
        )

        if n_devices > 1:
            V = V.sum()
            G = {i: G[i].sum() for i in G}

        return V, G

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

        n_devices = self.n_devices

        H = self._run_fun(
            self.hessian_batch,
            theta_train_dict,
            theta_nuisance_dict,
            data
        )

        if n_devices > 1:
            H = {i: {j: H[i][j].sum() for j in H[i]} for i in H}

        return H


