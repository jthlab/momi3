from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import tqdm

import math
import numdifftools as nd
from copy import deepcopy


class Experimental:
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
