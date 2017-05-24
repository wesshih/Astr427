'''
Wesley Shih
1237017
Astr 427 Homework 4
5/24/17
'''

import numpy as np

def golden(f, a, b, c):
	f_a = f(a)
	f_b = f(b)
	f_c = f(c)
	
	# make sure that the center point at b is "lower" than the outer
	# bounds
	assert(f_a > f_b)
	assert(f_c > f_b)

