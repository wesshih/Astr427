'''
Wesley Shih
1237017
Astr 427 Homework 4
5/24/17
'''

import numpy as np
import scipy.constants as sci

def golden(f, a, b, c):
	'''
	there must be a minimum on the interval a to b
	'''
	f_a = f(a)
	f_b = f(b)
	f_c = f(c)
	

	# make sure that the center point at b is "lower" than the outer
	# bounds
	assert(f_a > f_b)
	assert(f_c > f_b)

	# print 'a',a,'f_a',f_a
	# print 'b',b,'f_b',f_b
	# print 'c',c,'f_c',f_c

	iter = 0
	# while we are not at a final answer
	while not np.isclose(np.abs(c - a), 0.0):
		iter += 1
		left = (b - a) > (c - b) # true if the left portion is bigger
		dif = (c-a)/sci.golden
		x = c - dif if left else a + dif
		f_x = f(x)

		# lets always shoot for interval such that (a,b,x,c)
		if x < b:
			# this means we are in (a,x,b,c) so swap b and x
			b,x = x,b
			f_b,f_x = f_x,f_b

		# now pick a new interval to probe
		if (f_x > f_b):
			c,f_c = x,f_x
		else:
			a,f_a = b,f_b
			b,f_b = x,f_x

	print 'golden took',iter,'iterations'
	return ((a,f_a),(b,f_b),(c,f_c))

def main():
	print 'main test real quick'
	f = lambda x: x**2 - x
	a,b,c = 0.0, 2-sci.golden, 1.0
	print golden(f,a,b,c)

if __name__ == '__main__':
	main()