import sys
import os

ARGS = sys.argv[1:]

if ARGS == []:
	import jax
else:
	n_devices = int(ARGS[0])
	os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={n_devices}'
	import jax
	jax.config.update('jax_platform_name', 'cpu')


import demes
from momi3.MOMI import Momi


def test():
	n = 10
	demo = demes.load("tests/yaml_files/TwoDemes.yml")
	sampled_demes = ["A", "B"]
	sample_sizes = 2 * [n]
	momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)

	params = momi._default_params
	params.set_train_all_etas(True)
	params.set_train_all_taus(True)
	params.set_train_all_pis(True)

	jsfs = momi.simulate(1000, seed=108)

	with jax.debug_nans(True):
		momi.sfs_entry({'A': 1, 'B': 0})
		momi.sfs_spectrum()
		momi.loglik(params, jsfs)
		momi.loglik_with_gradient(params, jsfs)
		momi.GIM(params, jsfs)


if __name__ == "__main__":
	test()
