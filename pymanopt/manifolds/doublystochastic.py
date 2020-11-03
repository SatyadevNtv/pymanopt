import warnings

import numpy as np
from numpy import linalg as la, random as rnd
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, cg
import scipy
# Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
try:
    from scipy.linalg import solve_continuous_lyapunov as lyap
except ImportError:
    from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp

import sparse

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
    if hasattr(d1, 'fill_value'):
        d1.fill_value = np.float64(0)
    d2 = p / np.sum(C * d1[:, np.newaxis, :], axis=2)
    if hasattr(d1, 'fill_value'):
        d2.fill_value = np.float64(0)

    gap = np.inf


    while iters < maxiters:
        row = np.sum(C * d2[:, :, np.newaxis], axis=1)

        if iters % checkperiod == 0:
            try:
                if hasattr(row, 'multiply') or hasattr(d1, 'multiply'):
                    gap = np.max(np.absolute(d1.multiply(row) - q))
                else:
                    gap = np.max(np.absolute(row * d1 - q))
            except ValueError as e:
                embed()
            if np.any(np.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        if hasattr(d1, 'fill_value'):
            d1.fill_value = np.float64(0)
        d2 = p / np.sum(C * d1[:, np.newaxis, :], axis=2)
        if hasattr(d1, 'fill_value'):
            d2.fill_value = np.float64(0)

        if np.any(np.isnan(d1)) or np.any(np.isinf(d1)) or np.any(np.isnan(d2)) or np.any(np.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    _rhs_ = np.matmul(d2[:, :, np.newaxis], d1[:, np.newaxis, :])
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


        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)

        # `k` doublystochastic manifolds
        self._k = self._p.shape[0]

        self._name = ("{:d} {:d}X{:d} (max) matrices with positive entries such that row sum is p and column sum is q".format(self._k, n, m))

        self._dim = 0
        for i in range(self._k):
            self._dim += (self._p[i].nnz - 1) * (self._q[i].nnz - 1)

        self._e1 = scipy.sparse.csr_matrix(np.zeros((self._k, n)))
        self._e2 = scipy.sparse.csr_matrix(np.zeros((self._k, m)))
        for i in range(self._k):
            nnz_p = self._p[i].nnz
            nnz_q = self._q[i].nnz
            self._e1[i, :nnz_p] = np.ones(nnz_p)
            self._e2[i, :nnz_q] = np.ones(nnz_q)


    def __str__(self):
        return self._name


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        result = 0
        for i in range(self._k):
            result += (self._p[i].nnz + self._q[i].nnz)**2
        return np.sqrt(result)


    def inner(self, x, u, v):
        result = u*v/x
        result.fill_value = np.float64(0)
        return np.sum(result)


    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))


    def __prep_nd_sparse(self,):
        coords = [[], [], []]
        data = []
        for i in range(0, self._k):
            A0 = np.absolute(rnd.rand(self._p[i].nnz, self._q[i].nnz))
            A0 = A0[np.newaxis, :]
            nz = A0.nonzero()
            nz_val = A0[nz]
            coords[0] = coords[0] + (nz[0] + i).tolist()
            coords[1] = coords[1] + (nz[1]).tolist()
            coords[2] = coords[2] + (nz[2]).tolist()
            data += nz_val.tolist()
        result = sparse.COO(coords=coords, data=data, shape=(self._k, self._n, self._m))
        return result


    def rand(self):
        Z = self.__prep_nd_sparse()
        result = SKnopp(Z, self._p, self._q, self._maxSKnoppIters)
        return result


    def randvec(self, x):
        Z = self.__prep_nd_sparse()
        Zproj = self.proj(x, Z)
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        if sparse.COO(v).nnz == 0:
            return np.zeros(v.shape[0])

        vtop = scipy.sparse.csr_matrix(np.zeros((self._k, self._n)))
        vbottom = scipy.sparse.csr_matrix(np.zeros((self._k, self._m)))
        start_idx = 0
        for i in range(self._k):
            nnz_top = self._p[i].nnz
            nnz_bottom = self._q[i].nnz
            vtop[i, :nnz_top] = v[start_idx:start_idx+nnz_top]
            vbottom[i, :nnz_bottom] = v[start_idx+nnz_top:start_idx+nnz_top+nnz_bottom]
            start_idx += nnz_top+nnz_bottom

        Avtop = self._p.multiply(vtop) + np.sum(self.X * sparse.COO.from_scipy_sparse(vbottom)[:, np.newaxis, :], axis=2)
        Avbottom = np.sum(self.X * sparse.COO.from_scipy_sparse(vtop)[:, :, np.newaxis], axis=1) + self._q.multiply(vbottom)
        if np.any(np.isnan(Avtop)) or np.any(np.isnan(Avbottom)) or np.any(np.isinf(Avtop)) or np.any(np.isinf(Avbottom)):
            embed()
        result = np.array([])
        for i in range(self._k):
            result = np.concatenate((result, Avtop[i].data, Avbottom[i].data))
        return result


    def _lsolve(self, x, b):
        self.X = x.copy()
        shape = (len(b), len(b))
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), b, tol=1e-6, maxiter=100)
        if np.any(np.isnan(sol)):
            embed()
        del self.X
        start_idx = 0
        alpha = scipy.sparse.csr_matrix(np.zeros((self._k, self._n)))
        beta = scipy.sparse.csr_matrix(np.zeros((self._k, self._m)))
        for i in range(self._k):
            nnz_top = self._p[i].nnz
            nnz_bottom = self._q[i].nnz
            alpha[i, :nnz_top] = sol[start_idx:start_idx+nnz_top]
            beta[i, :nnz_bottom] = sol[start_idx+nnz_top:start_idx+nnz_top+nnz_bottom]
            start_idx += nnz_top+nnz_bottom
        return alpha, beta



    def proj(self, x, v):
        b = []
        b_n = np.sum(v, axis=2)
        b_m = np.sum(v, axis=1)
        for i in range(self._k):
            b += b_n[i].data.tolist() + b_m[i].data.tolist()
        alpha, beta = self._lsolve(x, np.array(b))
        result = v - (np.matmul(sparse.COO.from_scipy_sparse(alpha)[:,:,np.newaxis], sparse.COO.from_scipy_sparse(self._e2)[:, np.newaxis, :]) + \
                      np.matmul(sparse.COO.from_scipy_sparse(self._e1)[:, :, np.newaxis], sparse.COO.from_scipy_sparse(beta)[:, np.newaxis, :]))*x
        if np.any(np.isnan(result)):
            embed()
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        mu = x * u
        return self.proj(x, mu)


    def ehess2rhess(self, x, egrad, ehess, u):
        # TODO:
        raise RuntimeError
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
        Y = x * np.exp(u/x)
        Y.fill_value = np.float64(0)
        #Y = np.maximum(Y, 1e-50)
        #Y = np.minimum(Y, 1e50)
        res = SKnopp(Y, self._p, self._q, self._maxSKnoppIters)
        return res


    def zerovec(self, x):
        raise RuntimeError
        return np.zeros((self._n, self._m))


    def transp(self, x1, x2, d):
        return self.proj(x2, d)


