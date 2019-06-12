from sympy import *
from sympy.codegen.ast import Assignment

q = 9
d = 2

c = [ Matrix(x) for x in [(-1, 1), ( 0, 1), ( 1, 1), (-1, 0), ( 0, 0), ( 1, 0), (-1,-1), ( 0, -1), ( 1, -1)] ]
w = [ Rational(*x) for x in [(1,36), (1,9), (1,36), (1,9), (4,9), (1,9), (1,36), (1,9), (1,36)] ]

c_s = sqrt(Rational(1,3))

rho, tau = symbols('rho tau')

f_next = symarray('f_next', q)
f_curr = symarray('f_curr', q)

u = Matrix(symarray('u', d))

moments = [ Assignment(rho, sum(f_curr)) ]

for i, u_i in enumerate(u):
    moments.append(Assignment(u_i, sum([ (c_j*f_curr[j])[i] for j, c_j in enumerate(c) ]) / sum(f_curr)))

moments_opt = cse(moments, optimizations='basic', symbols=numbered_symbols(prefix='m'))

f_eq = []

for i, c_i in enumerate(c):
    f_eq_i = w[i] * rho * (  1
                           + c_i.dot(u)    /    c_s**2
                           + c_i.dot(u)**2 / (2*c_s**4)
                           - u.dot(u)      / (2*c_s**2) )
    f_eq.append(f_eq_i)

collide = [ Assignment(f_next[i], f_curr[i] + 1/tau * ( f_eq_i - f_curr[i] )) for i, f_eq_i in enumerate(f_eq) ]

collide_opt = cse(collide, optimizations='basic')
