import warnings

import numpy as np
from numpy import linalg as la, random as rnd
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, cg
# Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
try:
    from scipy.linalg import solve_continuous_lyapunov as lyap
except ImportError:
    from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp

import time
from IPython import embed


def SKnopp(A, p, q, maxiters=None, checkperiod=None):
    # TODO: Modify and optimize for same marginals

    tol = np.finfo(float).eps
    if maxiters is None:
        maxiters = A.shape[0]*A.shape[1]

    if checkperiod is None:
        checkperiod = 10

    if p.ndim < 2 and q.ndim < 2:
        p = p[np.newaxis, :]
        q = q[np.newaxis, :]

    C = A

    # TODO: Maybe improve this if-else by looking
    # for other broadcasting techniques
    if C.ndim < 3:
        d1 = q / np.sum(C, axis=0)[np.newaxis, :]
    else:
        d1 = q / np.sum(C, axis=1)

    if C.ndim < 3:
        d2 = p / np.einsum('nm,bm->bn', C, d1.conj())
    else:
        d2 = p / np.einsum('bnm,bm->bn', C, d1.conj())

    gap = np.inf

    iters = 0
    while iters < maxiters:
        if C.ndim < 3:
            row = np.einsum('bn,nm->bm', d2, C)
        else:
            row = np.einsum('bn,bnm->bm', d2, C)

        if iters % checkperiod == 0:
            gap = np.max(np.absolute(row * d1 - q))
            if np.any(np.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        if C.ndim < 3:
            d2 = p / np.einsum('nm,bm->bn', C, d1.conj())
        else:
            d2 = p / np.einsum('bnm,bm->bn', C, d1.conj())

        if np.any(np.isnan(d1)) or np.any(np.isinf(d1)) or np.any(np.isnan(d2)) or np.any(np.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    result = C * (np.einsum('bn,bm->bnm', d2, d1))
    return result


class DoublyStochastic(Manifold):
    """Manifold of `k` (n x m) positive matrices

    Implementation is based on multinomialdoublystochasticgeneralfactory.m
    """

    def __init__(self, n, m, p=None, q=None, maxSKnoppIters=None):
        self._n = n
        self._m = m
        self._p = np.array(p)
        self._q = np.array(q)
        self._maxSKnoppIters = maxSKnoppIters

        # Assuming that the problem is on single manifold.
        if p is None:
            self._p = np.repeat(1/n, n)
        if q is None:
            self._q = np.repeat(1/m, m)

        if self._p.ndim < 2 and self._q.ndim < 2:
            self._p = self._p[np.newaxis, :]
            self._q = self._q[np.newaxis, :]

        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)

        # `k` doublystochastic manifolds
        self._k = self._p.shape[0]

        self._name = ("{:d} {:d}X{:d} matrices with positive entries such that row sum is p and column sum is q respectively.".format(len(p), n, m))

        self._dim = self._k * (self._n - 1)*(self._m - 1)
        self._e1 = np.ones(n)
        self._e2 = np.ones(m)


    def __str__(self):
        return self._name


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        return np.sqrt(self._k) * (self._m + self._n)


    def inner(self, x, u, v):
        return np.sum(u * v/ x)


    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))


    def rand(self):
        Z = np.absolute(rnd.randn(self._n, self._m))
        return SKnopp(Z, self._p, self._q, self._maxSKnoppIters)


    def randvec(self, x):
        Z = rnd.randn(self._n, self._m)
        Zproj = self.proj(x, Z[np.newaxis, :, :])
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        self._k = int(self.X.shape[0])
        v = v.reshape(self._k, int(v.shape[0]/self._k))
        vtop = v[:, :self._n]
        vbottom = v[:, self._n:]
        Avtop = (vtop * self._p) + np.einsum('bnm,bm->bn', self.X, vbottom)
        Avbottom = np.einsum('bnm,bn->bm', self.X, vtop) + (vbottom * self._q)
        Av = np.hstack((Avtop, Avbottom))
        return Av.ravel()


    def _lsolve(self, x, b):
        self.X = x.copy()
        _dim = self._k * (self._n + self._m)
        shape = (_dim, _dim)
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), b, tol=1e-6, maxiter=1000)
        sol = sol.reshape(self._k, int(sol.shape[0]/self._k))
        del self.X
        alpha, beta = sol[:, :self._n], sol[:, self._n:]
        return alpha, beta



    def proj(self, x, v):
        assert v.ndim == 3
        b = np.hstack((np.sum(v, axis=2), np.sum(v, axis=1)))
        alpha, beta = self._lsolve(x, b.ravel())
        result = v - (np.einsum('bn,m->bnm', alpha, self._e2) + np.einsum('n,bm->bnm', self._e1, beta))*x
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        mu = x * u
        return self.proj(x, mu)


    def ehess2rhess(self, x, egrad, ehess, u):
        gamma = egrad * x
        gamma_dot = (ehess * x) + (egrad * u)

        assert gamma.ndim == 3 and gamma_dot.ndim == 3
        b = np.hstack((np.sum(gamma, axis=2), np.sum(gamma, axis=1)))
        b_dot = np.hstack((np.sum(gamma_dot, axis=2), np.sum(gamma_dot, axis=1)))

        alpha, beta = self._lsolve(x, b.ravel())
        alpha_dot, beta_dot = self._lsolve(
            x,
            b_dot.ravel() - np.hstack((
                np.einsum('bnm,bm->bn', u, beta),
                np.einsum('bnm,bn->bm', u, alpha)
            )).ravel()
        )

        S = np.einsum('bn,m->bnm', alpha, self._e2) + np.einsum('n,bm->bnm', self._e1, beta)
        S_dot = np.einsum('bn,m->bnm', alpha_dot, self._e2) + np.einsum('n,bm->bnm', self._e1, beta_dot)
        delta_dot = gamma_dot - (S_dot*x) - (S*u)

        delta = gamma - (S*x)

        nabla = delta_dot - (0.5 * (delta * u)/x)

        return self.proj(x, nabla)


    def retr(self, x, u):
        Y = x * np.exp(u/x)
        Y = np.maximum(Y, 1e-50)
        Y = np.minimum(Y, 1e50)
        res = SKnopp(Y, self._p, self._q, self._maxSKnoppIters)
        return res


    def zerovec(self, x):
        return np.zeros((self._k, self._n, self._m))


    def transp(self, x1, x2, d):
        return self.proj(x2, d)


