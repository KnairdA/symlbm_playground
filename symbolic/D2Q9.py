from sympy import *
from sympy.codegen.ast import Assignment

q = 9
d = 2

c = [ Matrix(x) for x in [(-1, 1), ( 0, 1), ( 1, 1), (-1, 0), ( 0, 0), ( 1, 0), (-1,-1), ( 0, -1), ( 1, -1)] ]
w = [ Rational(*x) for x in [(1,36), (1,9), (1,36), (1,9), (4,9), (1,9), (1,36), (1,9), (1,36)] ]

c_s = sqrt(Rational(1,3))

f_next = symarray('f_next', q)
f_curr = symarray('f_curr', q)

def moments(f = f_curr, optimize = True):
    rho = symbols('rho')
    u   = Matrix(symarray('u', d))

    exprs = [ Assignment(rho, sum(f)) ]

    for i, u_i in enumerate(u):
        exprs.append(Assignment(u_i, sum([ (c_j*f[j])[i] for j, c_j in enumerate(c) ]) / sum(f)))

    if optimize:
        return cse(exprs, optimizations='basic', symbols=numbered_symbols(prefix='m'))
    else:
        return ([], exprs)

def equilibrium():
    rho = symbols('rho')
    u   = Matrix(symarray('u', d))

    f_eq = []

    for i, c_i in enumerate(c):
        f_eq_i = w[i] * rho * (  1
                               + c_i.dot(u)    /    c_s**2
                               + c_i.dot(u)**2 / (2*c_s**4)
                               - u.dot(u)      / (2*c_s**2) )
        f_eq.append(f_eq_i)

    return f_eq

def bgk(tau, f_eq = equilibrium(), optimize = True):
    exprs = [ Assignment(f_next[i], f_curr[i] + 1/tau * ( f_eq_i - f_curr[i] )) for i, f_eq_i in enumerate(f_eq) ]

    if optimize:
        return cse(exprs, optimizations='basic')
    else:
        return ([], exprs)
