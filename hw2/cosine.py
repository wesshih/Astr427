'''
Wesley Shih
1237017
Astr 427 Homework 2
4/25/17

Problem 2: Diff Eq for cosine

In this problem we are asked to solve the differential equation of d^2x/dt^2 + x = 0.
We know this has an analytic solution of x(t) = cos(t), however we will use the functions
from problem 1 to solve this numerically.

As described in the integrate.py docstring, the following formats must be followed:

x = [x1, ... , xn, v1, ... , vn]
f has signature f(x, t)

'''

import numpy as np
import matplotlib.pyplot as plt
import integrate as myint

def f_cos(x, t):
	'''
	f_cos calculates the derivatie of a state vector x, such that it satisifies the
	the relationship specified by the differential equation.  Note that although the
	given ODE has time independent accel, our function f_cos must have a time parameter
	in order to meet the required function signature.

	Args:
		x: state vector to differentiate
		t: time to differentiate at

	Returns:
		state vector that is the derivative of x at time t
	'''
	mid = len(x)/2
	return np.concatenate((x[mid:], -x[:mid]))


def a():
	'''
	The purpose of part A is to use the integrator from problem 1 to successfully integrate
	the equation x''(t) + x(t) = 0.  I will use each of the stepper functions with a variety
	of step sizes, and compare how this affects the output of the integrator.
	'''

	# Let's initialize the initial conditions x(0) = 1.0, x'(0) = 0.0, 0.0 <= t <= 30.0,
	# and h is an element of [1.0, 0.3, 0.1, 0.03, 0.01]
	x0 = np.array([1.0, 0.0]) # x(0) = 1.0, x'(0) = 0.0
	ti, tf = 0.0, 100.0
	hs = np.array([1.0, 0.3, 0.1, 0.03, 0.01])

	# what about leap_stepper??
	steppers = [myint.euler_stepper, myint.mid_stepper, myint.rk_stepper]

	# make a figure for each type of stepper
	for step in steppers:
		# now run for various step size for this particular stepper
		color = iter(['r','b','g','c','m'])
		count = 511
		fig = plt.figure('Stepper: ' + `step.__name__`,figsize=(10,10))
		ls = []
		for h in hs:
			ts = myint.get_timesteps(ti, tf, h)
			res = myint.runner(step, f_cos, x0, ts, h)

			c = color.next()
			# plt.subplot(count).scatter(ts, res[:,0],alpha=0.5,s=20,c=temp_col)
			l = plt.subplot(count).plot(ts, res[:,0], 's-', alpha=0.5,c=c)
			ls.append((l[0],str('h:%0.3f'%h))) # for the legend

			# lets plot the known solution of x=cos(t) to compare
			# we should probably calculate this earlier so i don't have to keep doing it
			# just leave it for now
			# also need to make it move visible
			plt.subplot(count).plot(ts, np.cos(ts), ':',alpha=0.9, lw=2.0, c='k')

			# plt.subplot(count).set_xticklabels([])
			# plt.subplot(count).set_yticklabels([])
			# plt.subplot(count).set_ylim(-5,5)
			# plt.subplot(count).set_xlim(-1,31)
			count += 1

		# plt.subplots_adjust(wspace=0, hspace=0)
		# fig.tight_layout()

		ls = zip(*ls)
		fig.legend(ls[0], ls[1], 'upper center',mode='expand',ncol=2)
	
	plt.show()



def b():
	'''
	Part B asks us to find the difference between our numerical calculation and the exact value
	at the final time tf = 30.0.  The absolute value of the difference should be plotted against
	step size on a log-log plot.  By looking at the graph we will hopefully be able to draw 
	conclusions regarding the convergence of the error term.
	'''
	# First we will calculate and plot the error term. Analysis will come after.
	x0 = np.array([1.0, 0.0])
	ti, tf = 0.0, 30.0

	# array of various step sizes to plot, decide if you're going to do other than specified step size
	# hs = np.linspace(0.003, 0.3, 30)
	hs = np.array([1.0, 0.3, 0.1, 0.03, 0.01])

	
	err_e = np.array([])
	err_m = np.array([])
	err_r = np.array([])
	err_l = np.array([])

	for h in hs:
		ts = myint.get_timesteps(ti, tf, h)
		res_euler = myint.runner(myint.euler_stepper, f_cos, x0, ts, h)
		res_mid = myint.runner(myint.mid_stepper, f_cos, x0, ts, h)
		res_rk = myint.runner(myint.rk_stepper, f_cos, x0, ts, h)
		res_lf = myint.leap_runner(f_cos, x0, ts, h)

		err_e = np.append(err_e, np.abs(res_euler[-1,0] - np.cos(tf)))
		err_m = np.append(err_m, np.abs(res_mid[-1,0] - np.cos(tf)))
		err_r = np.append(err_r, np.abs(res_rk[-1,0] - np.cos(tf)))
		err_l = np.append(err_l, np.abs(res_lf[-1,0] - np.cos(tf)))

	fig = plt.figure("Part B", figsize=(10,10))
	le = plt.loglog(hs, err_e, 'rs-')[0]
	lm = plt.loglog(hs, err_m, 'gs-')[0]
	lr = plt.loglog(hs, err_r, 'bs-')[0]
	ll = plt.loglog(hs, err_l, 'ms-')[0]
	fig.legend((le, lm, lr, ll), ('euler', 'midpoint', 'RK4', 'Leap Frog'), 'upper right')
	plt.show()

	# First lets quickly describe what I'm doing here.  hs is an array of step-sizes that 
	# ranges from 0.003 to 0.3 in 30 equal steps.  So, in a nutshell, I have 30 step-sizes
	# to check between 0.003 and 0.3.  What we can see from this plot is that when the step
	# size is large, the error term is large.  This makes sense, as a large step size means
	# any error in our estimation of x at some time will be amplified by the large step size.
	# As step size gets smaller (around h = 0.1), the error actually drops below 1. This is
	# pretty good, but the error continues to improve as h approaches 0.01 and smaller. The 
	# error term is clearly trending towards 0, and it looks like it will converge with the
	# analytic solution with "infinitely" small step size.
	#
	# It is also interesting to note that although RK4 is the best by far, a step size of
	# h = 0.003 gives an error of less than 0.01 for all 4 algorithms.


def main():
	print 'cosine main'
	a()
	b()


if __name__ == '__main__':
	main()