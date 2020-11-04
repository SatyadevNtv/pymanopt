import warnings

import numpy as np
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

import cupy as cp
from cupy import random as rnd


def SKnopp(A, p, q, maxiters=None, checkperiod=None):
    # TODO: Modify and optimize for "same marginals" case

    A = cp.array(A)
    p = cp.array(p)
    q = cp.array(q)

    tol = cp.finfo(float).eps
    if maxiters is None:
        maxiters = A.shape[0]*A.shape[1]

    if checkperiod is None:
        checkperiod = 10

    if p.ndim < 2 and q.ndim < 2:
        p = p[cp.newaxis, :]
        q = q[cp.newaxis, :]

    C = A

    # TODO: Maybe improve this if-else by looking
    # for other broadcasting techniques
    if C.ndim < 3:
        d1 = q / cp.sum(C, axis=0)[cp.newaxis, :]
    else:
        d1 = q / cp.sum(C, axis=1)

    if C.ndim < 3:
        d2 = p / cp.einsum('nm,bm->bn', C, d1.conj(), dtype='float')
    else:
        d2 = p / cp.einsum('bnm,bm->bn', C, d1.conj(), dtype='float')

    gap = cp.inf

    iters = 0
    while iters < maxiters:
        if C.ndim < 3:
            row = cp.einsum('bn,nm->bm', d2, C, dtype='float')
        else:
            row = cp.einsum('bn,bnm->bm', d2, C, dtype='float')

        if iters % checkperiod == 0:
            gap = cp.max(cp.absolute(row * d1 - q))
            if cp.any(cp.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        if C.ndim < 3:
            d2 = p / cp.einsum('nm,bm->bn', C, d1.conj(), dtype='float')
        else:
            d2 = p / cp.einsum('bnm,bm->bn', C, d1.conj(), dtype='float')

        if cp.any(cp.isnan(d1)) or cp.any(cp.isinf(d1)) or cp.any(cp.isnan(d2)) or cp.any(cp.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    result = C * (cp.einsum('bn,bm->bnm', d2, d1, dtype='float'))
    return result


class DoublyStochastic(Manifold):
    """Manifold of `k` (n x m) positive matrices

    Implementation is based on multinomialdoublystochasticgeneralfactory.m
    """

    def __init__(self, n, m, p=None, q=None, maxSKnoppIters=None):
        self._n = n
        self._m = m
        self._p = cp.array(p)
        self._q = cp.array(q)
        self._maxSKnoppIters = maxSKnoppIters

        # Assuming that the problem is on single manifold.
        if p is None:
            self._p = cp.repeat(1/n, n)
        if q is None:
            self._q = cp.repeat(1/m, m)

        if self._p.ndim < 2 and self._q.ndim < 2:
            self._p = self._p[cp.newaxis, :]
            self._q = self._q[cp.newaxis, :]

        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)

        # `k` doublystochastic manifolds
        self._k = self._p.shape[0]

        self._name = ("{:d} {:d}X{:d} matrices with positive entries such that row sum is p and column sum is q respectively.".format(len(p), n, m))

        self._dim = self._k * (self._n - 1)*(self._m - 1)
        self._e1 = cp.ones(n)
        self._e2 = cp.ones(m)


    def __str__(self):
        return self._name


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        return cp.sqrt(self._k) * (self._m + self._n)


    def inner(self, x, u, v):
        x = cp.array(x)
        u = cp.array(u)
        v = cp.array(v)

        return cp.sum(u * v/ x).get()


    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))


    def rand(self):
        Z = cp.absolute(rnd.randn(self._n, self._m))
        return SKnopp(Z, self._p, self._q, self._maxSKnoppIters).get()


    def randvec(self, x):
        raise RuntimeError
        Z = rnd.randn(self._n, self._m)
        Zproj = self.proj(x, Z[cp.newaxis, :, :])
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        #start_time = time.time()
        #print(f"Entry .. {start_time}")
        self._k = int(self.X.shape[0])
        v = v.reshape(self._k, int(v.shape[0]/self._k))
        vtop = cp.array(v[:, :self._n])
        vbottom = cp.array(v[:, self._n:])
        Avtop = (vtop * self._p) + cp.einsum('bnm,bm->bn', self.X, vbottom, dtype='float')
        Avbottom = cp.einsum('bnm,bn->bm', self.X, vtop, dtype='float') + (vbottom * self._q)
        Av = cp.hstack((Avtop, Avbottom))
        result = Av.ravel().get()
        #end_time = time.time()
        #print(f"Exit. Time: {end_time}. Elapsed: {end_time - start_time}")
        return result


    def _lsolve(self, x, b):
        self.X = x.copy()
        _dim = self._k * (self._n + self._m)
        shape = (_dim, _dim)
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), b.get(), tol=1e-6, maxiter=100)
        sol = sol.reshape(self._k, int(sol.shape[0]/self._k))
        del self.X
        alpha, beta = sol[:, :self._n], sol[:, self._n:]
        return cp.array(alpha), cp.array(beta)



    def proj(self, x, v):
        assert v.ndim == 3
        b = cp.hstack((cp.sum(v, axis=2), cp.sum(v, axis=1)))
        alpha, beta = self._lsolve(x, b.ravel())
        result = v - (cp.einsum('bn,m->bnm', alpha, self._e2, dtype='float') + cp.einsum('n,bm->bnm', self._e1, beta, dtype='float'))*x
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        x = cp.array(x)
        u = cp.array(u)
        mu = x * u
        return self.proj(x, mu).get()


    def ehess2rhess(self, x, egrad, ehess, u):
        x = cp.array(x)
        egrad = cp.array(egrad)
        ehess = cp.array(u)

        gamma = egrad * x
        gamma_dot = (ehess * x) + (egrad * u)

        assert gamma.ndim == 3 and gamma_dot.ndim == 3
        b = cp.hstack((cp.sum(gamma, axis=2), cp.sum(gamma, axis=1)))
        b_dot = cp.hstack((cp.sum(gamma_dot, axis=2), cp.sum(gamma_dot, axis=1)))

        alpha, beta = self._lsolve(x, b.ravel())
        alpha_dot, beta_dot = self._lsolve(
            x,
            b_dot.ravel() - cp.hstack((
                cp.einsum('bnm,bm->bn', u, beta, dtype='float'),
                cp.einsum('bnm,bn->bm', u, alpha, dtype='float')
            )).ravel()
        )

        S = cp.einsum('bn,m->bnm', alpha, self._e2, dtype='float') + cp.einsum('n,bm->bnm', self._e1, beta, dtype='float')
        S_dot = cp.einsum('bn,m->bnm', alpha_dot, self._e2, dtype='float') + cp.einsum('n,bm->bnm', self._e1, beta_dot, dtype='float')
        delta_dot = gamma_dot - (S_dot*x) - (S*u)

        delta = gamma - (S*x)

        nabla = delta_dot - (0.5 * (delta * u)/x)

        return self.proj(x, nabla).get()


    def retr(self, x, u):
        x = cp.array(x)
        u = cp.array(u)

        Y = x * cp.exp(u/x)
        Y = cp.maximum(Y, 1e-50)
        Y = cp.minimum(Y, 1e50)
        res = SKnopp(Y, self._p, self._q, self._maxSKnoppIters)
        return res.get()


    def zerovec(self, x):
        return np.zeros((self._k, self._n, self._m))


    def transp(self, x1, x2, d):
        x2 = cp.array(x2)
        d = cp.array(d)
        return self.proj(x2, d).get()


