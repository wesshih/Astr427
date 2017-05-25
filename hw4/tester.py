'''
Wesley Shih
1237017
Astr 427 Homework 4
5/24/17
'''

import numpy as np
import golden as g
import scipy.constants as sci
import matplotlib.pyplot as plt

# f = lambda x: x**2 -x
# a,b,c = 0.0, 0.01, 1.5
# res = g.golden(f,a,b,c)
# print res

dat = np.loadtxt('rot.dat')
print dat

v_inf = 100.0
r_0 = 1.0
f = lambda r: v_inf * (1.0 - np.power(sci.golden, -r/r_0))
