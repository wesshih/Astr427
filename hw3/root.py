'''
Wesley Shih
1237017
Astr 427 Homework 3
4/28/17

This file contains the root finding methods of bisection and Newton-Ralphson.
Because in this assignment we are interested in how the successive approx
of the zeros converge, these methods return arrays of approximations.  If I 
was writing this as a real package, these methods would return only the final
result. For this assignment, the "final" result will be in the last array 
index.
'''

import numpy as np

def bisection(f, a, b):
	'''
	Find the root of function f between a and b using bisection

	Args:
		f: function that we can call to eval f at some x. must have funtion sig
			of f(x) that returns the value of the function at x.
		a,b: the endpoints of the region we will look for a root in.
			Should have f(a) < 0 and f(b) > 0

	Returns:
		array of (x_i,f(x_i)) where x_i is the ith approx of the root between
			a and b.
	'''

	# keep track of these values to prevent from recalculating things
	f_a, f_b = f(np.array([a,b]))

	# make sure a root exists between a and b
	if not f_a*f_b < 0.0:
		return None

	if f_a > f_b:
		# swap a and b
		a, f_a, b, f_b = b, f_b, a, f_a

	res = np.array([[0,0]]) # need initial value, will throw away later

	# set max number of iterations of 1000 to prevent looping for eternity
	for i in xrange(1000):
		m = (a + b) * 0.5
		f_m  = f(m)
		res = np.append(res, [[m, f_m]], axis=0)
		if np.isclose(f_m, 0.0):
			break
		if f_m*f_a < 0.0:
			b = m
			f_b = f_m
		else:
			a = m
			f_a = f_m

	return res[1:]


def newton(f, df, x0):
	'''
	Find the root of function f between a and b using newton method.

	Args:
		f: function that we can call to eval f at some x. must have funtion sig
			of f(x) that returns the value of the function at x.
		df: function that we can call to evaluate the derivative of f at some x
			Must have the function sig df(x) that returns the value of the 
			derivative at x
		x0: Initial x to start the newton method at

	Returns:
		array of (x_i,f(x_i)) where x_i is the ith approx of the root between
	'''

	res = np.array([[0, 0]]) # need initial value, will throw away later

	# loop for at max 1000 times
	for i in xrange(1000): 
		if np.isclose(df(x0), 0.0):
			# if the derivative is 0 at x0 then we cannot continue
			break
	
		res = np.append(res, [[x0, f(x0)]], axis=0)
		if np.isclose(res[-1,1], 0.0):
			break
		else:

			x0 -= res[-1,1]/df(x0)

	return res[1:]
