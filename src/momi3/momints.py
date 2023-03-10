"momints: moments meets momi!"

import itertools as it

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps


def _downsample(n) -> sps.coo_array:
    """Phi[n-1,i-1] in terms of Phi[n]:

        (M Phi[n])[i-1] = Phi[n-1,i-1], i=1,...,n+1

    Returns:
        The array M shown above.
    """
    i = np.arange(1, n + 1)
    return sps.diags([n - i + 1, i], [0, 1], shape=(n, n + 1)) / n


def _drift(n):
    "drift operator for a population of size n + 1"
    i = jnp.arange(n + 1)
    return sps.diags(
        [((i - 1) * (n - i + 1))[1:], -2 * i * (n - i), ((n - i - 1) * (i + 1))[:-1]],
        [-1, 0, 1],
    )


def _migration(ns):
    """construct migration matrix for population u migrating into v, when the overall sizes of the
    populations are given by ns"""
    # (A12) in the manuscript:
    #
    #     M Phi[n+ej](i) = nk(ij+1) / (nj+1) * [Phi[n+ej-ek](i+ej-ek) - Phi[n+ej-ek](i+ej)]
    #                      - ik Phi[n](i) + (ik + 1)Phi[n](i+ek)
    #
    assert len(ns) == 2
    [1 + n for n in ns]
    nj, nk = ns
    # first construct Phi[n + ej - ek] by jacknife and downsample.
    # shift the j axis forward by one, because each term is indexed as `i + ej`, and multiply the result j axis by d
    J = _jackknife(nj).tocsr()[1:]
    # leading coefficient, (1 + ij) * nk / (1 + nj), acts along columns
    d = (1 + np.arange(nj + 1)) * nk / (nj + 1)
    Phi0 = sps.diags(d) @ J
    # similarly, shift when we downsample; the probability of getting nk in the downsampled pop is 0
    Phi1 = sps.vstack([_downsample(nk), np.zeros(nk + 1)])
    # Phi1[ij,ik] now equals Phi[n + ej - ek](i + ej) for ij=0,...,nj and ik=0,...,nk
    # now we shift the k axis down by one to reach Phi[n + ej - ek](i + ej - ek)
    Phi2 = sps.vstack([np.zeros([1, nk + 1]), Phi1.tocsc()[:-1, :]])
    # this is the term in square brackets in (A12), premultiplied by nk(ij+1)/(nj+1)
    ret1 = {0: Phi0, 1: Phi2 - Phi1}
    # other two terms, first up is -ik * Phi[n]
    Phi3 = sps.diags(np.arange(nk + 1))
    # the last term is (ik + 1) * Phi[n](i+ek) = (ik + 1) * Sek
    Phi4 = sps.diags(np.arange(1, nk + 1), 1)
    ret2 = {1: Phi4 - Phi3}
    return ret1, ret2


def _mutation(n):
    "mutation operator"
    i = np.arange(n + 1)
    return sps.diags(
        [(n - i + 1)[1:], -n, (i + 1)[:-1]], [-1, 0, 1], shape=(n + 1, n + 1)
    )


def _ibis(n):
    """find the best approximation of i' / (n + 1) for 2 <= i <= n - 2"""
    i = np.arange(1, n + 1)
    # this should be as simple as:
    #
    #     np.argmin(abs(i[1:-2, None] / n - i[None] / (1 + n)), axis = 0)
    #
    # but: the function used in moments rounds to the nearest odd integer in the case of ties for whatever reason.
    # have to do some contortions to match that...
    y = i * n / (n + 1)
    z = np.rint(y)
    c = np.isclose(y - 0.5, np.floor(y))
    z[c] = np.ceil(y[c])
    return z.clip(2, n - 2)
    # return np.array([moments.Jackknife.index_bis(i + 1, n) for i in range(n)])


def _jackknife0(n):
    """jackknife0 operator expresses "interior terms" Phi_{n+1}, 1=,...,n in terms of Phi_{n}"""
    i = np.arange(n)
    ibis = _ibis(n) - 1
    row, col = np.concatenate([[i, ibis - 1], [i, ibis], [i, ibis + 1]], axis=1)
    Q_beta = (
        -(1.0 + n)
        * (
            (2.0 + i) * (2.0 + n) * (-6.0 - n + (i + 1.0) * (3.0 + n))
            - 2.0 * (4.0 + n) * (-1.0 + (i + 1.0) * (2.0 + n)) * (ibis + 1.0)
            + (12.0 + 7.0 * n + n**2) * (ibis + 1.0) ** 2
        )
        / (2.0 + n)
        / (3.0 + n)
        / (4.0 + n)
    )
    Q_alpha = (
        (1.0 + n)
        * (
            4.0
            + (1.0 + i) ** 2 * (6.0 + 5.0 * n + n**2)
            - (i + 1.0) * (14.0 + 9.0 * n + n**2)
            - (4.0 + n) * (-5.0 - n + 2.0 * (i + 1.0) * (2.0 + n)) * (ibis + 1.0)
            + (12.0 + 7.0 * n + n**2) * (ibis + 1.0) ** 2
        )
        / (2.0 + n)
        / (3.0 + n)
        / (4.0 + n)
        / 2.0
    )
    Q_gamma = (
        (1.0 + n)
        * (
            (2.0 + i) * (2.0 + n) * (-2.0 + (i + 1.0) * (3.0 + n))
            - (4.0 + n) * (1.0 + n + 2.0 * (i + 1.0) * (2.0 + n)) * (ibis + 1.0)
            + (12.0 + 7.0 * n + n**2) * (ibis + 1.0) ** 2
        )
        / (2.0 + n)
        / (3.0 + n)
        / (4.0 + n)
        / 2.0
    )
    data = np.concatenate([Q_alpha, Q_beta, Q_gamma])
    return sps.coo_matrix((data, (row, col)), shape=(n, n - 1))


def _jackknife(n):
    J0 = _jackknife0(n).tocsr()
    # from the downsampling formula, Phi[n+1,0] = Phi[n,0] - Phi[n+1,1]/(n+1)
    #                                Phi[n+1,n+1] = Phi[n,n] - Phi[n+1,n]/(n+1)
    # augment the jackknife matrix to also handle the cases i=0, i=n
    return sps.bmat(
        [
            [1.0, -J0[0] / (n + 1), None],
            [None, J0, None],
            [None, -J0[-1] / (n + 1), 1.0],
        ]
    )


def _Qm_matrix(dims, M):
    """migration rate matrix for different pops (testing usage only)"""
    ret = {}
    for (i, di), (j, dj) in it.combinations(enumerate(dims), 2):
        u1, u2 = _migration([di - 1, dj - 1])
        v1, v2 = _migration([dj - 1, di - 1])
        ret[(i, j)] = M[i, j] * sps.kron(u1[0], u1[1])
        ret[(i, j)] += M[j, i] * sps.kron(v1[1], v1[0])
        ret[(i, j)] += sps.kronsum(M[i, j] * u2[1], M[j, i] * v2[1])
    return ret
