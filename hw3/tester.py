'''
Wesley Shih
1237017
Astr 427 Homework 3
4/28/17

This file contains functions that test root finding functions in root.py

'''

import root
import numpy as np
import matplotlib.pyplot as plt
import time

def parabolic_test():
	# f(x) = x^2 - a
	f_par = lambda x: x**2 - 2.0
	df_par = lambda x: 2*x
	
	# now lets set up the bracket such that on (a,b), f(a)*f(b)<0 and f(a)<f(b)
	# lets pick a = 0, and b = 4
	zs_b = root.bisection(f_par, 0.0, 4.0) # zeros for bisection
	zs_n = root.newton(f_par, df_par, 2.0) # zeros for newton

	print 'parabolic test:'
	print '\t','bisection found zero at x =',zs_b[-1,0],'in',len(zs_b),'steps'
	print '\t','newton found zero at x =',zs_n[-1,0],'in',len(zs_n),'steps'

	plot_zeros(f_par, 0.9, 2.1, zs_b, zs_n)

	'''
	Discussion: Both methods find the root of ~ 1.414. This makes sense, as
	we are essentially solving what is sqrt(2.0). However, we see that the 
	newton method finds the root significantly faster, as it takes 5 steps
	compared to the 29 steps the bisection method takes. We see that the
	bisection method's successive approx of the root ping-pong between
	the left and right of the root, whereas the newton method goes directly
	to the zero. This will not always be the case, as we will see in kepler.
	'''

def kepler_test():
	# M = E - e*sinE
	f_kep = lambda E: M + e*np.sin(E) - E
	df_kep = lambda E: e*np.cos(E) - 1
	
	# First find zeros of M=1.5 and e=0.5
	M = 1.5
	e = 0.5

	zs_b = root.bisection(f_kep, 0.0, 2*np.pi)
	zs_n = root.newton(f_kep, df_kep, 1.0)

	print 'Kepler test with e = 0.5:'
	print '\t','bisection found zero at x =',zs_b[-1,0],'in',len(zs_b),'steps'
	print '\t','newton found zero at x =',zs_n[-1,0],'in',len(zs_n),'steps'
	
	plot_zeros(f_kep, 0.0, 2*np.pi, zs_b, zs_n)

	# now find zeros with e = 0.9
	e = 0.9
	zs_b = root.bisection(f_kep, 0.0, 2*np.pi)
	zs_n = root.newton(f_kep, df_kep, 1.0)

	print 'Kepler test with e = 0.9:'
	print '\t','bisection found zero at x =',zs_b[-1,0],'in',len(zs_b),'steps'
	print '\t','newton found zero at x =',zs_n[-1,0],'in',len(zs_n),'steps'

	plot_zeros(f_kep, 0.0, 2*np.pi, zs_b, zs_n)

	'''
	Discussion: As in the parabolic test, we see that although both methods
	find roots that are essentially equal, the newton's method finds the zero
	substantially faster than the bisection method.  Unlike the parabolic
	function, both methods 'ping-pong' between x values that are to the left
	and the right of the zero.  However, by using derivative information we
	can close in on the root substantially faster.

	It is worth pointing out that if we choose the initial guess for newton
	poorly, then we can end up in a situation where our successive guesses
	take us far from the actual root. In this case we will still find the 
	correct zero eventually, however it will take more iterations. 
	Uncomment the following code to see this behavior.
	'''
	
	# zs_n = root.newton(f_kep, df_kep, 0.1)
	# plot_zeros(f_kep, 0.0, 2*np.pi, zs_b, zs_n)


def plot_zeros(f, x0, xf, zs_b, zs_n):
	'''
	convenience function to plot the zero approx against the actual function

	Args:
		f: function that we wish to plot
		x0, xf: the x range that we wish to plot over
		zs_b: the zeros as found by the bisection method
		zs_n: the zeros as found by the newton method
	'''
	# plot the successive approx of zeros on top of the actual function
	xs = np.linspace(x0, xf, 20)
	ys = f(xs)

	# subplot for bisection method
	plt.figure(figsize=(10,10))
	plt.subplot(211).plot(zs_b[:,0], zs_b[:,1],'o-', c='r')
	plt.subplot(211).plot(xs, ys, 'k') # plot actual function
	plt.subplot(211).plot(xs, [0]*len(xs), 'k') # plot x-axis

	# subplot for newton metho2
	plt.subplot(212).plot(zs_n[:,0], zs_n[:,1], 's-', c='b')
	plt.subplot(212).plot(xs, ys, 'k') # plot actual function
	plt.subplot(212).plot(xs, [0]*len(xs), 'k') # plot x-axis

	plt.show()

def fast_kepler():
	'''
	Problem 2: find the root of the kepler problem for various values of M,
	and compare how quickly the bisection and newton methods solve it.
	'''
	f_kep = lambda E: M + e*np.sin(E) - E
	df_kep = lambda E: e*np.cos(E) - 1
	
	Ms = np.linspace(0.0, 2*np.pi, 20)
	e = 0.5

	time_b, time_n = [], []
	zs_b, zs_n  = [], []
	plt.figure(figsize=(10,10))
	# Find the root for each value of M
	for M in Ms:
		# first run bisection and time it
		start = time.time()
		zs_b.append(root.bisection(f_kep, -0.1, 2*np.pi + 0.1)[-1,0])
		end = time.time()
		time_b.append(end-start)

		# now do the same for newton
		start = time.time()
		zs_n.append(root.newton(f_kep, df_kep, 1.0)[-1,0])
		end = time.time()
		time_n.append(end-start)

	print 'bisection timing:'
	print '\t','avg:',np.average(time_b)
	print '\t','tot:',np.sum(time_b)

	print 'newton timing:'
	print '\t','avg:',np.average(time_n)
	print '\t','tot:',np.sum(time_n)

	plt.scatter(np.cos(zs_n), np.sin(zs_n), c='r')
	plt.show()

	'''
	Discussion: In terms of execution time, we see that the netwon method
	is always faster than the bisection method. It is faster both for the
	average time to find a zero, and for the total time to find all zeros.
	This makes sense as the newton method requires significantly fewer 
	iterations to find the zeros.  As a whole, the newton's method takes
	roughly 1/3 of the time that the bisection method takes.

	The second thing that we can see from this problem is how the actual
	orbit is behaving. In this situation, the body in motion is orbiting
	aroung the sun, which is at a point with some 0 < x < 1. We see that
	for equal time steps, the oribiting body "covers more space" when close
	to the sun (at the perihelion), and "covers less space" when far away.
	This makes total sense as the large kinetic energy that the body has
	when close to the sun is converted to potential as it is "flung" away.
	This is something that I knew and expected, however, it is still pretty
	cool to see it come out of my own calculation.
	'''



def main():
	'''
	Runs all of the testing functions. Note that because these tests plot
	their results, the plot of some test must be closed before the next
	test can be run.
	'''
	parabolic_test()
	print ''
	kepler_test()
	print ''
	fast_kepler()

if __name__ == '__main__':
	main()