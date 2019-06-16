from sympy import *

q = 27
d = 3

c = [ Matrix(x) for x in [
    (-1, 1, 1), ( 0, 1, 1), ( 1, 1, 1), (-1, 0, 1), ( 0, 0, 1), ( 1, 0, 1), (-1,-1, 1), ( 0, -1, 1), ( 1, -1, 1),
    (-1, 1, 0), ( 0, 1, 0), ( 1, 1, 0), (-1, 0, 0), ( 0, 0, 0), ( 1, 0, 0), (-1,-1, 0), ( 0, -1, 0), ( 1, -1, 0),
    (-1, 1,-1), ( 0, 1,-1), ( 1, 1,-1), (-1, 0,-1), ( 0, 0,-1), ( 1, 0,-1), (-1,-1,-1), ( 0, -1,-1), ( 1, -1,-1)
]]

w = [Rational(*x) for x in [
    (1, 216), (1,54), (1,216), (1,54), (2,27), (1,54), (1,216), (1,54), (1,216),
    (1,  54), (2,27), (1, 54), (2,27), (8,27), (2,27), (1, 54), (2,27), (1, 54),
    (1, 216), (1,54), (1,216), (1,54), (2,27), (1,54), (1,216), (1,54), (1,216)
]]

c_s = sqrt(Rational(1,3))
