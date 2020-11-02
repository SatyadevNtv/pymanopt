from numpy import random as rnd
import numpy as np
from scipy.io import loadmat

from pymanopt.manifolds.doublystochastic import DoublyStochastic, SKnopp
from pymanopt.solvers import ConjugateGradient, TrustRegions
from pymanopt import Problem
from pymanopt import function

from IPython import embed

def test_doublystochastic():
    rnd.seed(21)

    ns = [100, 100] * 10
    ms = [200, 200] * 10
    batch = len(ns)

    p = []
    q = []
    A = []
    for i in range(batch):
        n, m = ns[i], ms[i]
        p0 = np.random.rand(n)
        q0 = np.random.rand(m)
        p.append(p0 / np.sum(p0))
        q.append(q0 / np.sum(q0))
        A0 = rnd.rand(n, m)
        A0 = A0[np.newaxis, :]
        A0 = SKnopp(A0, p[i], q[i], n+m)
        A.append(A0)
    A = np.vstack((C for C in A))

    def _cost(x):
        return 0.5 * (np.linalg.norm(x - A)**2)

    def _egrad(x):
        return x - A

    def _ehess(x, u):
        return u

    manf = DoublyStochastic(n, m, p, q)
    solver = TrustRegions(maxiter=50, maxtime=100000)
    prblm = Problem(manifold=manf, cost=lambda x: _cost(x), egrad=lambda x: _egrad(x), ehess=lambda x, u: _ehess(x, u), verbosity=3)

    U = manf.rand()
    Uopt = solver.solve(prblm, x=U)
    for i in range(batch):
        print(f"""
    For {i}
    Row constraint err: {np.linalg.norm(np.sum(Uopt[i], axis=0) - q[i])}
    Col constraint err: {np.linalg.norm(np.sum(Uopt[i], axis=1) - p[i])}

        """)


if __name__ == "__main__":
    test_doublystochastic()
