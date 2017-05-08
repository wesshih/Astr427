'''
Wesley Shih
1237017
Astr 427 Homework 3
4/28/17


'''

import numpy as np
import matplotlib.pyplot as plt

class func:
	def __init__(self, f, df):
		self.f = f
		self.df = df


def bisection(f, a, b):
	'''
	Find the root of function f between a and b

	Args:
		f: functor that we can call to eval f at some x. must have funtion sig
			of f(x) that returns the value of the function at x.
		a,b: the endpoints of the region we will look for a root in.
			Should have f(a) < 0 and f(b) > 0

	Returns:
		the x value of the root between a and b if it exists. Returns None otherwise
	'''

	# keep track of these values to prevent from recalculating things
	f_a, f_b = f((a,b))

	if not f_a*f_b < 0.0:
		return None

	if f_a > f_b:
		# swap a and b and recall to have f(a) < f(b)
		a, f_a, b, f_b = b, f_b, a, f_a
		# return bisection(f, b, a)

	for i in xrange(1000): # set max number of iterations so we don't loop for eternity
		m = (a + b) * 0.5
		f_m  = f(m)
		if np.isclose(f_m, 0.0):
			return m

		if f_m*f_a < 0.0:
			b = m
			f_b = f_m
		else:
			a = m
			f_a = f_m

	# if got here, then we must have exceeded max number of iterations
	return None


def newton(f, df, x0):
	for i in xrange(1000): # don't expect to hit this max, but it does provide safety
		if np.isclose(df(x0), 0.0):
			return None
	
		if np.isclose(f(x0), 0.0):
			return x0
		else:
			x0 -= f(x0)/df(x0)


def bisection_step(f, a, b):
	# because the bisection runner (or maybe just genaric runner) will be calling this,
	# we'll assume that the arguments are properly formatted, and that a step is necessary

	# returns a tuple with the updated bracket. either (a,m,done) or (m,b,done) depending on f(m)
	# the last element of the tuple is a boolean that describes if we found an m such that f(m) = 0
	# if done == 1, then both of the other two entries will hold m.
	m = (a + b) * 0.5
	f_m = f(m)
	if np.isclose(f_m, 0.0):
		return (m, m, True)
	return (a,m,0) if f_m*f(a) < 0.0 else (m,b,0)

def bisection_runner(f, a, b):
	# calculate the original end points of the bracket
	f_a, f_b = f((a, b))

	if not f_a*f_b < 0.0:
		# bracket (and therefore besection method) cannot ensure that a zero exists
		return None

	if f_a > f_b:
		# swap a and b to make things work out below
		a,b = b,a
		f_a,f_b = f_b,f_a

	for i in xrange(1000):
		a,b,done = bisection_step(f, a, b)
		if done:
			return a
	# reached the end of a kajillion loop executions, return None
	return None

# print df_kep(0)
# x = np.linspace(0.0, 2*np.pi, 100)
# y = f_kep(x)
# dy = df_kep(x)
# # plt.scatter(x, y)
# # plt.scatter(x, dy, c='r')

# e = 0.2
# xs = np.cos(x)-e
# ys = e*np.sin(x)
# plt.scatter(xs, ys, c='g')

# plt.show()

# def test(x):
# 	return np.power(x, 2)

# es = np.array([1,2,3])
# print 'es',es
# es2 = f_kep(es)
# print 'es2',es2
# es3 = f_kep((5,6,10,12))
# print es3
# print ''
# print type(es),type(es2),type(es3)