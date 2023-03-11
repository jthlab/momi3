from momi3 import esfs


def test_esfs(iwm):
    sample_sizes = {"deme0": 5, "deme1": 3}
    e = esfs(iwm, sample_sizes)
    assert e.shape == (6, 4)
