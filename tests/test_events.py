import itertools as it

import numpy as np
import stdpopsim

import momi3.momi
from momi3.sfs.events import Lift


def test_mig_bug1():
    demo = (
        stdpopsim.get_species("HomSap")
        .get_demographic_model("OutOfAfrica_3G09")
        .model.to_demes()
    )
    mm = demo.migration_matrices()
    mm[1].insert(0, np.inf)
    ax = [d.name for d in demo.demes]
    params = demo.asdict()
    m3 = momi3.momi.Momi3(demo).iicr({"YRI": 2})
    T = m3._T
    for edge in T.edges():
        e = T.edges[edge].get("event")
        if isinstance(e, Lift):
            M, _ = e.migration_matrix(params, ax)
            t0 = e.t0.t
            t1 = e.t1.t if np.isfinite(e.t1.t) else 2 * t0
            for t in np.linspace(t0, t1, 100):
                M_true = next(
                    MM
                    for MM, (t1, t0) in zip(mm[0], it.pairwise(mm[1]))
                    if t0 <= t < t1
                )
                M_true = np.array(M_true)
                M_true -= np.diag(np.sum(M_true, axis=1))
                np.testing.assert_allclose(M_true, M(t))
