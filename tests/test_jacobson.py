import demes
import pytest

from momi3.momi import Momi3


def get_momi(yaml_path, n=4, jitted=True, batch_size=1000):
    demo = demes.load(yaml_path / "jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi3(demo, sampled_demes, sample_sizes, jitted=True, batch_size=batch_size)
    return momi


@pytest.mark.slow
def test_branch_length(yaml_path):
    # Load demes graph
    momi = get_momi(yaml_path)
    return momi.total_branch_length()


def test_loglik(yaml_path, nmuts=5, batch_size=5):
    momi = get_momi(yaml_path, batch_size=batch_size)
    jsfs = momi.simulate(nmuts, seed=108)
    print(f"non-zero-entries={jsfs.nnz}")
    val, c_time, r_time = momi._time_loglik(momi._default_params, jsfs=jsfs, repeat=5)
    print(f"loglik: {val}")
    print(f"Compilation time: {c_time}")
    print(f"Runtime: {r_time}")


def test_bounds(yaml_path, n=20):
    demo = demes.load(yaml_path / "jacobson.yml")
    demo_dict = demo.asdict()
    new_zero = 100  # If a migration ends at time 0, replace it with 100
    for mig in demo_dict["migrations"]:
        if mig["end_time"] == 0:
            mig["end_time"] = new_zero
    demo = demes.Builder.fromdict(demo_dict).resolve()

    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 9 * [n]
    momi = Momi3(demo, sampled_demes, sample_sizes, jitted=False)
    momi.bound_sampler(momi._default_params, [], 100)
    # KeyError: Node(i=-31, block=frozenset({'Den1', 'Nea1', 'Papuan', 'Ghost', 'YRI', 'CHB'}), t=Time(1412.0))


if __name__ == "__main__":
    from pathlib import Path

    test_branch_length(yaml_path=Path("tests/yaml_files"))
    # args = sys.argv[1:]
    # if args[0] == "loglik":
    #     # python test_jacobson.py loglik <nmuts> <batch_size>
    #     nmuts = int(args[1])
    #     batch_size = int(args[2])
    #     test_loglik(nmuts, batch_size)
    # test_branch_length()
