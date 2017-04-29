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


def bisection(func, a, b):
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
	f_a = func.f(a)
	f_b = func.f(b)
	f_m = None

	if not f_a*f_b < 0.0:
		return None

	if f_a > f_b:
		# recalc will a < b
		return bisection(func, b, a)

	max_iter = 10000 # make sure we don't loop for eternity
	while max_iter > 0:
		max_iter -= 1
		m = (a + b) * 0.5
		f_m  = func.f(m)
		if np.isclose(f_m, 0.0):
			return m
		if f_m*f_a < 0.0:
			b = m
			f_b = f_m
		else:
			a = m
			f_a = f_m
	# if got here, then we must have exceeded max_iter
	return None
	# this works, but has the potential for recalculating a lot of stuff we don't need to
	# if np.isclose(f(m), 0):
	# 	return m
	# else:
	# 	return bisection(f, m, b) if f(m) < 0 else bisection(f, a, m)

def newton(func, x0):
	if np.isclose(func.f(x0), 0.0):
		return x0

	if np.isclose(func.df(x0), 0.0):
		return None

	delta = func.f(x0)/func.df(x0)
	return newton(func, x0-delta)


f_test = func(lambda x: np.power(x, 2) - 4.0, lambda x: 2*x)
print bisection(f_test, 0, 4)
print newton(f_test, 0)


def f_kep(E):
	M = 1.5
	e = 0.5
	return M + e*np.sin(E) - E

ff = func(f_kep, lambda x: 0)

print bisection(ff, 0, 2*np.pi)

x = np.linspace(0.0, 2*np.pi, 20)
y = f_kep(x)
plt.scatter(x, y)
plt.show()