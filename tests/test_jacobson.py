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


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "loglik":
        # python test_jacobson.py loglik <nmuts> <batch_size>
        nmuts = int(args[1])
        batch_size = int(args[2])
        test_loglik(nmuts, batch_size)
    # test_branch_length()
