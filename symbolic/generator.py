from sympy import *
from sympy.codegen.ast import Assignment

class LBM:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.f_next = symarray('f_next', descriptor.q)
        self.f_curr = symarray('f_curr', descriptor.q)

    def moments(self, optimize = True):
        rho = symbols('rho')
        u   = Matrix(symarray('u', self.descriptor.d))

        exprs = [ Assignment(rho, sum(self.f_curr)) ]

        for i, u_i in enumerate(u):
            exprs.append(
                Assignment(u_i, sum([ (c_j*self.f_curr[j])[i] for j, c_j in enumerate(self.descriptor.c) ]) / sum(self.f_curr)))

        if optimize:
            return cse(exprs, optimizations='basic', symbols=numbered_symbols(prefix='m'))
        else:
            return ([], exprs)

    def equilibrium(self):
        rho = symbols('rho')
        u   = Matrix(symarray('u', self.descriptor.d))

        f_eq = []

        for i, c_i in enumerate(self.descriptor.c):
            f_eq_i = self.descriptor.w[i] * rho * ( 1
                                                  + c_i.dot(u)    /    self.descriptor.c_s**2
                                                  + c_i.dot(u)**2 / (2*self.descriptor.c_s**4)
                                                  - u.dot(u)      / (2*self.descriptor.c_s**2) )
            f_eq.append(f_eq_i)

        return f_eq

    def bgk(self, tau, f_eq, optimize = True):
        exprs = [ Assignment(self.f_next[i], self.f_curr[i] + 1/tau * (f_eq_i - self.f_curr[i])) for i, f_eq_i in enumerate(f_eq) ]

        if optimize:
            return cse(exprs, optimizations='basic')
        else:
            return ([], exprs)
