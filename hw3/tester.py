import root
import numpy as np

# Definitions for the functions and derivatives used problem 1. We will refer to
# the parabolic function as f_par/df_par, and the kepler functions as f_kep/df_kep

def parabolic_test():
	# f(x) = x^2 - a
	a = 2.0
	f_par = lambda x: x**2 - a
	df_par = lambda x: 2*x
	
	# now lets set up the bracket such that on (a,b), f(a)*f(b) < 0 and f(a) < f(b)
	# lets pick a = 0, and b = 4
	z = root.bisection(f_par, 0.0, 4.0)
	print z

def test(x):
  return x**2 - 2.0

def kepler_test():
	# M = E - e*sinE
	M = 1.5
	e = 0.5
	f_kep = lambda E: M + e*np.sin(E) - E
	df_kep = lambda E: e*np.cos(E) - 1
	
	x = np.linspace(0.0, 2*np.pi, 100)
	y = f_kep(x)
	#dy = 


parabolic_test()
