import jax
jax.config.update('jax_platform_name', 'cpu')

import numpy as np

from momi3.utils import update
from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.Data import get_X_batches

import timeit
import demes

import momi as momi2

from copy import deepcopy


def model(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))
    g_NEA = np.log(PD['eta_6'] / PD['eta_5']) / (PD['tau_11'] - PD['tau_6'])

    # Momi 2 model building
    model1 = momi2.DemographicModel(N_e=PD["eta_0"], muts_per_gen=None)

    for i in range(16):
        if i == 5:
            pass
        else:
            eta = f"eta_{i}"
            model1.add_size_param(eta, PD[eta])

    for i in range(13):
        tau = f"tau_{i}"
        model1.add_size_param(tau, PD[tau])

    model1.add_growth_param("g_NEA", g_NEA)
    
    model1.add_pulse_param("pi_2", PD["pi_2"])
    model1.add_pulse_param("pi_1", PD["pi_1"])
    model1.add_pulse_param("pi_0", PD["pi_0"])
   
    model1.add_leaf('Mbuti', N='eta_8')
    model1.add_leaf('Han', N='eta_11')
    model1.add_leaf('Sardinian', N='eta_15')
    model1.add_leaf('Loschbour', N='eta_3')
    model1.add_leaf('LBK', N='eta_14')
    model1.add_leaf('MA1', N="eta_12")
    model1.add_leaf('UstIshim', N='eta_10')
    model1.add_leaf('Neanderthal', N='eta_7')

    model1.set_size('BasalEurasian', N='eta_9', t='tau_0')
    
    model1.move_lineages("Sardinian", "Loschbour", t='tau_1', p='pi_2')
    model1.move_lineages("Sardinian", "LBK", t="tau_2")
    
    model1.set_size('LBK', N='eta_13', t='tau_2')
    
    model1.move_lineages("LBK", "BasalEurasian", t="tau_3", p='pi_1')
    model1.move_lineages("LBK", "Loschbour", t='tau_4')
    model1.move_lineages("MA1", "Loschbour", t='tau_5')

    model1.set_size('Neanderthal', N='eta_6', t='tau_6', g='g_NEA')

    model1.move_lineages("Han", "Loschbour", t="tau_7")
    
    model1.set_size('Loschbour', N='eta_2', t='tau_7')
    
    model1.move_lineages("UstIshim", "Loschbour", t="tau_8")
    model1.move_lineages("Loschbour", "Neanderthal", t="tau_9", p='pi_0')
    model1.move_lineages("BasalEurasian", "Loschbour", t="tau_10")
    model1.move_lineages("Mbuti", "Loschbour", t="tau_11")

    model1.set_size('Neanderthal', N='eta_4', t='tau_11')
    model1.set_size('Loschbour', N='eta_1', t='tau_11')

    model1.move_lineages("Neanderthal", "Loschbour", t="tau_12")

    model1.set_size('Loschbour', N='eta_0', t='tau_12')

    # momi2.DemographyPlot(model1, ['p1', 'p2', 'p3', 'p4', 'p5'])

    model1._mem_chunk_size = 10000

    return model1


def get_demo():
	return demes.load('yaml_files/8_pop_3_admix.yaml')


if __name__ == "__main__":
    # python tests/time_5_pop_admixture.py <method> <sample size> <number of positions> <number of replications> <save folder>
    # e.g. python tests/time_8_pop_admixture.py momi2 4 100 10 /tmp/
    demo = get_demo()
    sampled_demes = demo.metadata['sampled_demes']
    sample_sizes = demo.metadata['sample_sizes']
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)

    params = Params(momi)
    params.set_train_all(True)

    _f = momi._T.execute

    theta_dict = params._theta_path_dict

    def esfs_map(theta_dict, X, auxd, demo, _f):
        # X[pop].shape = (A, B)
        # A: jax.vmap size
        # B: sample size + 1

        def esfs_tensor_prod(X):
            demo_dict = demo.asdict()
            for paths, val in theta_dict.items():
                for path in paths:
                    update(demo_dict, path, val)
            return _f(demo_dict, X, auxd)

        return jax.vmap(esfs_tensor_prod, 0)(X)
    esfs_map = jax.jit(esfs_map, static_argnames=('_f'))

    SEQLEN = [1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]  # seqlen for chromosome sim
    JSFS = {}
    for seqlen in SEQLEN:
        jsfs = momi.simulate_chromosome(int(seqlen), 1e-8, 1e-8, seed=100)
        JSFS[seqlen] = jsfs

    # momi 3 times:
    momitimes = {}
    momivals = {}
    for seqlen in SEQLEN:
        jsfs = JSFS[seqlen].copy()
        deriveds = tuple(tuple(i) for i in jsfs.nonzero())

        def f():
            X = get_X_batches(momi.sampled_demes, momi.sample_sizes, tuple(momi._T.leaves), deriveds, add_etbl_vecs=False)
            X = deepcopy(X)

            for pop in X:
                X[pop] = X[pop][0][0]

            return esfs_map(theta_dict, X, momi._T.auxd, demo, _f)

        momivals[seqlen] = f()
        t = timeit.repeat(f, number=1, repeat=10)
        momitimes[seqlen] = np.median(t)

    # momi 2 times
    PD = dict(zip(params._keys, params._theta))
    momi2_model = lambda theta_train: model(params._theta_train, train_keys=params._train_keys, PD=PD)
    num_sample = dict(zip(sampled_demes, sample_sizes))

    def momi2_model_func(x):
        return momi2_model(x)._get_demo(num_sample)
    m2m = momi2_model(params._theta_train)

    momi2times = {}
    momi2vals = {}

    for seqlen in SEQLEN:
        jsfs = JSFS[seqlen]

        data = np.array(jsfs.data)
        coords = np.array(jsfs.coords.T)
        n = [i - 1 for i in jsfs.shape]
        P = len(n)
        config_list = {
            tuple((n[j] - coord[j], coord[j]) for j in range(P)): val for coord, val in zip(coords, data)
        }

        sfs = momi2.site_freq_spectrum(sampled_demes, [config_list])

        def f():
            m2m.set_data(sfs, length=1)
            m2m.set_mut_rate(1.0)
            return m2m.expected_sfs()

        momi2vals[seqlen] = f()
        t = timeit.repeat(f, number=1, repeat=10)
        momi2times[seqlen] = np.median(t)

    for seqlen in SEQLEN:
        print(f'non zero entries={int(JSFS[seqlen].nnz)}')
        print(f'momi2 runtime: {momi2times[seqlen]:.3g}')
        print(f'momi3 runtime: {momitimes[seqlen]:.3g}')
        print(10 * '==')
