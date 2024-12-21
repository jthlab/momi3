import numpy as np

import momi3.momi
import demes
from momi3.common import get_path


def test_reparam():
    # isolation with interrupted migration model
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=1e4)])
    b.add_deme(
        "A",
        epochs=[dict(start_size=1e4, end_time=5e3), dict(start_size=1e3, end_time=0)],
        ancestors=["anc"],
        start_time=1e4,
    )
    b.add_deme("B", epochs=[dict(start_size=1e4)], ancestors=["anc"], start_time=1e4)
    b.add_migration(source="A", dest="B", rate=1e-4)
    graph = b.resolve()
    m3 = momi3.momi.Momi3(graph)
    et = m3.sfs({"A": 2, "B": 3})._T
    params = graph.asdict()
    paths = (
        ("demes", 1, "epochs", 0, "end_time"),
        ("demes", 2, "start_time"),
        ("demes", 1, "proportions"),
        ("migrations", 0, "rate"),
    )
    f, finv = et.reparameterize(paths)
    y = finv(params)
    params1 = f(y, params)
    for path in paths:
        np.testing.assert_allclose(get_path(params, path), get_path(params1, path))
