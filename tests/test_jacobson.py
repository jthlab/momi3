import sys

import demes

from momi3.MOMI import Momi


def get_momi(n=4, jitted=True, batch_size=1000):
    demo = demes.load("yaml_files/jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True, batch_size=batch_size)
    return momi


def test_branch_length():
    # Load demes graph
    momi = get_momi()
    return momi.total_branch_length()


def test_loglik(nmuts, batch_size):
    momi = get_momi(batch_size=batch_size)
    jsfs = momi.simulate(nmuts, seed=108)
    print(f"non-zero-entries={jsfs.nnz}")
    val, c_time, r_time = momi._time_loglik(momi._default_params, jsfs=jsfs, repeat=5)
    print(f"loglik: {val}")
    print(f"Compilation time: {c_time}")
    print(f"Runtime: {r_time}")


def test_bounds(n=20):
    demo = demes.load("yaml_files/jacobson.yml")
    demo_dict = demo.asdict()
    new_zero = 100  # If a migration ends at time 0, replace it with 100
    for mig in demo_dict['migrations']:
        if mig['end_time'] == 0:
            mig['end_time'] = new_zero
    demo = demes.Builder.fromdict(demo_dict).resolve()

    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi(demo, sampled_demes, sample_sizes, jitted=False)
    bounds = momi.bound_sampler(momi._default_params, [], 100)
    momi_b = Momi(demo, sampled_demes, sample_sizes, jitted=True, bounds=bounds)
    # KeyError: Node(i=-31, block=frozenset({'Den1', 'Nea1', 'Papuan', 'Ghost', 'YRI', 'CHB'}), t=Time(1412.0))


if __name__ == "__main__":
    test_branch_length()
    # args = sys.argv[1:]
    # if args[0] == "loglik":
    #     # python test_jacobson.py loglik <nmuts> <batch_size>
    #     nmuts = int(args[1])
    #     batch_size = int(args[2])
    #     test_loglik(nmuts, batch_size)
    # test_branch_length()
