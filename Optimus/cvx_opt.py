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
    r = matrix([-1.0, -1.0, -3.0])
    V = matrix(np.diag([1.0, 1.0, 1.0]))

    G = matrix(np.diag([-1.0,-1.0,-1.0]))
    h = matrix([0.33, 0.33, 0.33])

    A = matrix(np.ones(3)).T
    b = matrix(0.0, (1, 1))

    def F(x=None, z=None):
        if x is None:
            return 1, matrix([0.0,0.0,0.0])
        f = x.T * V * x - sigma2
        Df = x.T * (V + V.T)
        if z is None:
            return f, Df
        return f, Df, z[0,0] * (V+V.T)

    dims = {'l': h.size[0], 'q': [], 's': []}
    sol = solvers.cpl(r, F, G, h, dims, A, b)
    return sol['x']


if __name__ == '__main__':
    from cvxopt import matrix, log, div, spdiag, solvers

    def F(x=None, z=None):
        if x is None:  return 0, matrix(0.0, (3, 1))
        if max(abs(x)) >= 1.0:  return None
        u = 1 - x ** 2
        val = -sum(log(u))
        Df = div(2 * x, u).T
        if z is None:  return val, Df
        H = spdiag(2 * z[0] * div(1 + u ** 2, u ** 2))
        return val, Df, H


    G = matrix([[0., -1., 0., 0., -21., -11., 0., -11., 10., 8., 0., 8., 5.],
                [0., 0., -1., 0., 0., 10., 16., 10., -10., -10., 16., -10., 3.],
                [0., 0., 0., -1., -5., 2., -17., 2., -6., 8., -17., -7., 6.]])
    h = matrix([1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])
    dims = {'l': 0, 'q': [4], 's': [3]}
    sol = solvers.cp(F, G, h, dims)
    print(sol['x'])

    print(model2(3))
