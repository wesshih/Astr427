'''
Wesley Shih
1237017
Astr 427 Homework 4
5/24/17

This file contains functions that will optimize a given 1-D problem.
The only function included at this time is the golden search algo.
I had originally intended to play around with other optimizations,
but ran out of time.  I'll keep this file named optimize.py in case
I ever come back to it.
'''

import numpy as np
import scipy.constants as sci

def golden(f, a, b, c):
	'''
	Finds the minimum of the given function in the given bracket,
	using an optimal line search algorithm.

	Requires that a minimum of f lies in the interval [a,c], and 
	that f(b) < f(a),f(c)

	Args:
		f:	function to be optimized. must be 1-D
		a:	start point of search interval
		b:	point in (a,c) such that f(b) < f(a),f(c)
		c:	end point of search interval
	
	Returns:
		the minimum point (x_min, f(x_min))
	'''
	# assert that we were passed floats
	assert(type(a) is float)
	assert(type(b) is float)
	assert(type(c) is float)

	# function evaluations that we will keep between iterations
	f_a = f(a)
	f_b = f(b)
	f_c = f(c)

	# make sure starting bracket is okay
	assert(b > a)
	assert(c > b)
	assert(f_a > f_b)
	assert(f_c > f_b)

	# while we are not at a final answer
	# while not np.isclose(np.abs(c - a), 0.0):
	while not np.isclose(a, c):
		left = (b - a) > (c - b) # is the left portion of bracket bigger?
		dif = (c-a)/sci.golden
		x = c - dif if left else a + dif
		f_x = f(x)

		# want interval in form a < b < x < c
		if x < b:
			# swap b and x
			b,x = x,b
			f_b,f_x = f_x,f_b

		# now pick a new interval to probe
		if (f_x > f_b):
			# search interval (a,b,x)
			c,f_c = x,f_x
		else:
			# search interval (b,x,c)
			a,f_a = b,f_b
			b,f_b = x,f_x

	# We have found the minimum between the original a and c.
	# At this point np.isclose(a,c) == True, so essentiall a = c
	# Can use either as the minimum
	return (a, f(a))

