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

    tol = np.finfo(float).eps
    if maxiters is None:
        maxiters = A.shape[0]*A.shape[1]

    if checkperiod is None:
        checkperiod = 10

    if p.ndim < 2 and q.ndim < 2:
        p = p[np.newaxis, :]
        q = q[np.newaxis, :]

    if A.ndim < 3:
        A = A[np.newaxis, :]


    C = A

    iters = 0
    d1 = q / np.sum(A, axis=1)
    #d2 = p / np.einsum('bnm,bm->bn', C, d1.conj())
    d2 = p / np.tensordot(C, d1.conj(), axes=[2, 1])[:, :, 0]

    gap = np.inf

    while iters < maxiters:
        #row = np.einsum('bn,bnm->bm', d2, C)
        row = np.tensordot(d2, C, axes=[1, 1])[0]

        if iters % checkperiod == 0:
            gap = np.max(np.absolute(row * d1 - q))
            if np.any(np.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        #d2 = p / np.einsum('bnm,bm->bn', C, d1.conj())
        d2 = p / np.tensordot(C, d1.conj(), axes=[2, 1])[:, :, 0]

        if np.any(np.isnan(d1)) or np.any(np.isinf(d1)) or np.any(np.isnan(d2)) or np.any(np.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    _rhs_ = np.matmul(d2[:, :, np.newaxis], d1[:, np.newaxis, :])
    #result = C * (np.einsum('bn,bm->bnm', d2, d1))
    result = C * _rhs_
    return result


class DoublyStochastic(Manifold):
    """Manifold of (n x m) positive matrices

    Implementation is based on multinomialdoublystochasticgeneralfactory.m
    """

    def __init__(self, n, m, p=None, q=None, maxSKnoppIters=None):
        self._n = n
        self._m = m
        self._p = p
        self._q = q
        self._maxSKnoppIters = maxSKnoppIters

        if p is None:
            self._p = np.repeat(1/n, n)
        if q is None:
            self._q = np.repeat(1/m, m)

        if self._p.ndim < 2 and self._q.ndim < 2:
            self._p = self._p[np.newaxis, :]
            self._q = self._q[np.newaxis, :]


        # TODO: assert the the max_m, max_n (nonzeros) match (m, n)
        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)

        self._name = ("{:d}X{:d} matrices with positive entries such that row sum is p and column sum is q".format(n, m))
        # TODO: Make it appropriate
        self._dim = (n - 1)*(m - 1)
        self._e1 = np.ones(n)
        self._e2 = np.ones(m)


    def __str__(self):
        return self._name


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        # TODO:
        return self._m + self._n


    def inner(self, x, u, v):
        # TODO:
        return np.sum(u*v/x)


    def norm(self, x, u):
        # TODO:
        return np.sqrt(self.inner(x, u, u))


    def rand(self):
        # TODO:
        Z = np.absolute(rnd.randn(self._n, self._m))
        return SKnopp(Z, self._p, self._q, self._maxSKnoppIters)


    def randvec(self, x):
        # TODO:
        Z = rnd.randn(self._n, self._m)
        Zproj = self.proj(x, Z)
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        # TODO:
        batch = int(self.X.shape[0])
        v = v.reshape(batch, int(v.shape[0]/batch))
        vtop = v[:, :self._n]
        vbottom = v[:, self._n:]
        #Avtop = (vtop * self._p) + np.einsum('bnm,bm->bn', self.X, vbottom)
        Avtop = (vtop * self._p) + np.tensordot(self.X, vbottom, axes=[2,1])[:, :, 0]
        #Avbottom = np.einsum('bnm,bn->bm', self.X, vtop) + (vbottom * self._q)
        Avbottom = np.tensordot(self.X, vtop, axes=[1,1])[:, :, 0] + (vbottom * self._q)
        Av = np.hstack((Avtop, Avbottom))
        return Av.ravel()


    def _lsolve(self, x, b):
        # TODO:
        self.X = x.copy()
        batch = x.shape[0]
        _dim = batch * (self._n + self._m)
        shape = (_dim, _dim)
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), b, tol=1e-6, maxiter=100)
        sol = sol.reshape(batch, int(sol.shape[0]/batch))
        del self.X
        alpha, beta = sol[:, :self._n], sol[:, self._n:]
        return alpha, beta



    def proj(self, x, v):
        # TODO: Reverify this
        b = np.hstack((np.sum(v, axis=2), np.sum(v, axis=1)))
        alpha, beta = self._lsolve(x, b.ravel())
        #result = v - (np.einsum('bn,m->bnm', alpha, self._e2) + np.einsum('n,bm->bnm', self._e1, beta))*x
        result = v - (np.matmul(alpha[:,:,np.newaxis], self._e2[np.newaxis, :]) + \
                      np.matmul(self._e1[:, np.newaxis], beta[:, np.newaxis, :]))*x
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        # TODO:
        mu = x * u
        return self.proj(x, mu)


    def ehess2rhess(self, x, egrad, ehess, u):
        # TODO:
        gamma = egrad * x
        gamma_dot = (ehess * x) + (egrad * u)

        b = np.concatenate((np.sum(gamma, axis=1), np.sum(gamma, axis=0)))
        b_dot = np.concatenate((np.sum(gamma_dot, axis=1), np.sum(gamma_dot, axis=0)))

        alpha, beta = self._lsolve(x, b)
        alpha_dot, beta_dot = self._lsolve(
            x,
            b_dot - np.concatenate((
                np.matmul(u, beta),
                np.matmul(u.T, alpha)
            ))
        )

        S = np.outer(alpha, self._e2) + np.outer(self._e1, beta)
        S_dot = np.outer(alpha_dot, self._e2) + np.outer(self._e1, beta_dot)
        delta_dot = gamma_dot - (S_dot*x) - (S*u)

        delta = gamma - (S*x)

        nabla = delta_dot - (0.5 * (delta * u)/x)

        return self.proj(x, nabla)


    def retr(self, x, u):
        # TODO:
        #print(f"Retraction called ...")
        start_time = time.time()
        Y = x * np.exp(u/x)
        Y = np.maximum(Y, 1e-50)
        Y = np.minimum(Y, 1e50)
        res = SKnopp(Y, self._p, self._q, self._maxSKnoppIters)
        end_time = time.time()
        #print(f"Retraction time -> {end_time - start_time}")
        return res


    def zerovec(self, x):
        # TODO:
        return np.zeros((self._n, self._m))


    def transp(self, x1, x2, d):
        # TODO:
        return self.proj(x2, d)


