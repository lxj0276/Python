from cvxopt import matrix, solvers
import numpy as np


def model1():
    Q = 2 * matrix([[2, .5], [.5, 1]])
    print("Q:")
    print(Q)
    p = matrix([1.0, 1.0])
    print("p:")
    print(p)
    G = matrix([[-1.0, 0.0], [0.0, -1.0]])
    print("G:")
    print(G)
    h = matrix([0.0, 0.0])
    print("h:")
    print(h)
    A = matrix([1.0, 1.0], (1, 2))
    print("A:")
    print(A)
    b = matrix(1.0)
    print("b:")
    print(b)

    sol = solvers.qp(Q, p, G, h, A, b)
    print(sol['x'])
    print(sol['primal objective'])


def model2(sigma2):
    r = matrix([-1.0, -1.0, -1.0])
    V = matrix(np.ones(9).reshape(3,3))

    G = matrix(np.diag([-1.0,-1.0,-1.0]))
    h = matrix([0.0, 0.0, 0.0])

    A = matrix(np.ones(3)).T
    b = matrix(1.0, (1, 1))

    def F(x=None, z=None):
        if x is None:
            return 1, matrix([1.0,1.0,1.0])
        if min(x) <= 0.0:
            return None
        f = x.T * V * x - sigma2
        Df = x.T * (V + V.T)
        if z is None:
            return f, Df
        return f, Df, z[0,0] * (V+V.T)

    dims = {'l': h.size[0], 'q': [], 's': []}
    sol = solvers.cpl(r, F, G, h, dims, A, b)
    return sol['x']


if __name__ == '__main__':
    print(model2(3))