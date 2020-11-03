import sys
from numpy import random as rnd
import numpy as np
import scipy
from scipy.io import loadmat
import sparse

from pymanopt.manifolds.doublystochastic import DoublyStochastic, SKnopp
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt import Problem
from pymanopt import function

from IPython import embed

def test_doublystochastic():
    rnd.seed(21)

    ns = [100, 200]
    ms = [30, 400]
    m_max = np.array(ms).max()
    n_max = np.array(ns).max()
    batch = len(ns)

    p = []
    q = []
    A = []
    spr_p = scipy.sparse.csr_matrix(np.zeros((batch, n_max)))
    spr_q = scipy.sparse.csr_matrix(np.zeros((batch, m_max)))

    for i in range(0, batch):
        n, m = ns[i], ms[i]
        p0 = np.ones(n)
        q0 = np.ones(m)
        p.append(p0 / np.sum(p0))
        q.append(q0 / np.sum(q0))
        A0 = rnd.rand(n, m)
        A0 = A0[np.newaxis, :]
        A0 = SKnopp(A0, p[i], q[i], n + m)
        A.append(A0)
        spr_p[i, :n] = p[i]
        spr_q[i, :m] = q[i]
    coords = [[], [], []]
    data = []
    for i in range(0, batch):
        nz = A[i].nonzero()
        nz_val = A[i][nz]
        coords[0] = coords[0] + (nz[0] + i).tolist()
        coords[1] = coords[1] + (nz[1]).tolist()
        coords[2] = coords[2] + (nz[2]).tolist()
        data += nz_val.tolist()
    spr_A = sparse.COO(coords=coords, data=data, shape=(batch, n_max, m_max))


    def _cost(x):
        return 0.5 * (np.sum((x - spr_A)**2))

    def _egrad(x):
        return x - spr_A

    def _ehess(x, u):
        raise RuntimeError
        return u

    manf = DoublyStochastic(n_max, m_max, spr_p, spr_q)
    solver = ConjugateGradient(maxiter=50, maxtime=100000)
    prblm = Problem(manifold=manf, cost=lambda x: _cost(x), egrad=lambda x: _egrad(x), ehess=lambda x, u: _ehess(x, u), verbosity=3)

    U = manf.rand()
    Uopt = solver.solve(prblm, x=U)
    print(f"""
For all
Row constraint err: {np.sum((np.sum(Uopt, axis=1) - spr_q)**2)}
Col constraint err: {np.sum((np.sum(Uopt, axis=2) - spr_p)**2)}

    """)


if __name__ == "__main__":
    test_doublystochastic()
