import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

        self.m = b.shape[0]
        self.A = matvec_Ax(np.eye(self.m))

    def func(self, x):
        # np.sum() и так хорошо работает, не надо умножать вектор на единичный вектор - это тоже самое!
        # func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2
        return 1/m * np.sum(np.logaddexp(0, -np.diag(self.b) @ self.matvec_ATx(x))) + self.regcoef / 2 * np.linalg.norm(x)**2

    def grad(self, x):
        # grad(x) = -(1 - sigma(-diag(b) * Ax)) * b * A + regcoef * x
        # предположим, что сигма - диагогальная матрица
        sigma = np.diag(scipy.special.expit(-np.diag(self.b) @ self.matvec_ATx(x)))
        return - (1 - sigma) * self.b * self.A + self.regcoef * x

    def hess(self, x):
        # предположим, что сигма - диагогальная матрица
        sigma = np.diag(scipy.special.expit(-np.diag(self.b) @ self.matvec_ATx(x)))
        return sigma * (1 - sigma) * np.linalg.norm(self.A)**2 + self.refcoef * np.eye(self.m)


# class LogRegL2OptimizedOracle(LogRegL2Oracle):
#     """
#     Oracle for logistic regression with l2 regularization
#     with optimized *_directional methods (are used in line_search).
#
#     For explanation see LogRegL2Oracle.
#     """
#     def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
#         super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
#
#     def func_directional(self, x, d, alpha):
#         # TODO: Implement optimized version with pre-computation of Ax and Ad
#         return None
#
#     def grad_directional(self, x, d, alpha):
#         # TODO: Implement optimized version with pre-computation of Ax and Ad
#         return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    ------------------------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A
    """
    matvec_Ax = lambda x: np.dot(A, x)  # TODO: Implement
    matvec_ATx = lambda x: np.dot(A.T, x)  # TODO: Implement
    matmat_ATsA = lambda s: np.dot(A.T, np.dot(np.diag(s), A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    # Отмечено желтым цветом в файлике
    # elif oracle_type == 'optimized':
    #     oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    return (func(x + eps) - func(x)) / eps

def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    E = np.eye(n)
    res = np.zeros((n, n))
    for i in range(n)
        for j in range(i, n):
            res[i, j] = (func(x + eps * E[i] * E[j]) - func(x + eps * E[i]) - func(x + eps * E[j]) + func(x)) / eps**2
            res[j, i] = res[i, j]
    return res

if __name__ == '__main__':
    # A - m x n
    A = np.array([[1, 2], [-4, 3], [5, 7]])
    b = np.array([1, -1, 1])
    oracle = create_log_reg_oracle(A, b, regcoef, oracle_type='usual')
    func = lambda x: oracle.func(x)
    x = np.array([2, 3])
    print(oracle.grad(x))
    print(grad_finite_diff(func, x))