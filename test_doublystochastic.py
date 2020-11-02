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

    n = 1001
    m = 2000

    p0 = 2 * np.ones(n)
    q0 = 2 * np.ones(m)
    p = p0 / np.sum(p0)
    q = q0 / np.sum(q0)

    A0 = rnd.rand(n, m)
    A0 = A0[np.newaxis, :]
    A0 = SKnopp(A0, p, q, n+m)
    A = A0 + 1e-3 * rnd.rand(n, m)[np.newaxis, :]

    def _cost(x):
        return 0.5 * (np.linalg.norm(x - A)**2)

    def _egrad(x):
        return x - A

    def _ehess(x, u):
        raise RuntimeError
        return u

    p_batch = np.array([p, p])
    q_batch = np.array([q, q])
    manf = DoublyStochastic(n, m, p_batch, q_batch)
    #manf = DoublyStochastic(n, m, p, q)
    solver = ConjugateGradient(maxiter=50, maxtime=100000)
    prblm = Problem(manifold=manf, cost=lambda x: _cost(x), egrad=lambda x: _egrad(x), ehess=lambda x, u: _ehess(x, u), verbosity=3)

    U = manf.rand()
    Uopt = solver.solve(prblm, x=U)
    print(f"""
Row constraint err: {np.linalg.norm(np.sum(Uopt[0], axis=0) - q)}
Col constraint err: {np.linalg.norm(np.sum(Uopt[0], axis=1) - p)}
    """)


if __name__ == "__main__":
    test_doublystochastic()
