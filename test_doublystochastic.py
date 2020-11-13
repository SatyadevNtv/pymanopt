import cProfile
import sys
from numpy import random as rnd
import numpy as np
import cupy as cp
from scipy.io import loadmat
from scipy.sparse import csr_matrix, linalg as splg

from pymanopt.manifolds.doublystochastic import DoublyStochastic, SKnopp, SparseSKnopp
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt import Problem
from pymanopt import function

from IPython import embed

def test_doublystochastic(N, M, K):
    rnd.seed(21)

    ns = [N] * K
    ms = [M] * K
    batch = len(ns)

    # Sparse support.
    # 0's point to the constraint
    H = csr_matrix(np.eye(N, M))

    p = []
    q = []
    A = None
    for i in range(batch):
        n, m = ns[i], ms[i]
        p0 = np.random.rand(n)
        q0 = np.random.rand(m)
        p.append(p0 / np.sum(p0))
        q.append(q0 / np.sum(q0))
        A0 = rnd.rand(n, m)
        A0 = SparseSKnopp(H.multiply(A0), p[i], q[i], n+m)
        A = A0

    def _cost(x):
        result = 0.5 * (splg.norm(np.array(x) - np.array(A))**2)
        return result

    def _egrad(x):
        return x - A

    def _ehess(x, u):
        return u

    manf = DoublyStochastic(n, m, p, q, spr_mask=H)
    solver = ConjugateGradient(maxiter=3, maxtime=100000)
    prblm = Problem(manifold=manf, cost=lambda x: _cost(x), egrad=lambda x: _egrad(x), ehess=lambda x, u: _ehess(x, u), verbosity=3)

    U = manf.rand()
    Uopt = np.array(solver.solve(prblm, x=U).todense())
    print(f"""
For all
Row constraint err: {np.linalg.norm(np.sum(Uopt, axis=0) - q[0])}
Col constraint err: {np.linalg.norm(np.sum(Uopt, axis=1) - p[0])}

    """)


if __name__ == "__main__":
    n, m, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    test_doublystochastic(n, m, k)
