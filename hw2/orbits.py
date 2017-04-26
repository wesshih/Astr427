'''
Wesley Shih
1237017
Astr 427 Homework 2
4/25/17

Problem 3: 2-D Orbits

This problem asks us to use our ODE integrator to solve the 2D orbit described by
a potential Phi = -1/(1 + x^2 + y^2)^1/2. This is fairly similar to a 1/r potential,
so we expect the solutions to look more or less elliptical.  Because it is not exactly
1/r, we will have an ellipse that precesses around the origin.

As described in the integrate.py docstring, the following formats must be followed:

x = [x1, ... , xn, v1, ... , vn]
f has signature f(x, t)

'''
import numpy as np
import matplotlib.pyplot as plt
import integrate as myint

# simply assign the second half of the array to be the first half -> dxi/dt = vi
# then set dvidt = -xi/(1 + x1^2 + ... + xn^2)^(3/2)
def f_orbit(x, t):
	'''
	f_orbit calculates the derivative of a state vector x, such that it satisfies the
	relationship given by the differential equation.  Now in 2 spatial dimensions, we have
	4 coupled equations that we must solve.  Note that both accelerations are independent
	of time, but like f_cos, we need to include t as a parameter for  function signature.

	Args:
		x: state vector to differentiate
		t: specified time to differentiate at
	
	Results:
		state vector that is the derivative of x at time t
	'''

	# TODO: decide what you want to do in terms of readability vs concision. Also look at
	# runtime and make sure that you pick the more efficient one.

	# m = len(x)/2 # m for mid-index of array
	# denom = np.power(1 + np.sum(np.power(x[:m], 2)), 1.5)
	# return np.concatenate((x[m:], -x[:m]/denom))
	mid = len(x)/2
	return np.concatenate((x[mid:], -x[:mid]/np.power(1.0 + np.sum(np.power(x[:mid], 2)), 1.5)))



def a():
	'''
	Part A asks us to integrate the differential equation that describes a 2D orbit
	for 0.0 <= t <= 100.0 with initial conditions x(0) = 1, y(0) = 0, x'(0) = 0, y'(0) = 0.3.
	We will use Leapfrog and RK4 to perform this calculation, and maybe use the other two
	just to see how different they are. We'll run this calculation several times with various
	step sizes and look at how sensitive the output is to different step sizes.

	'''

	# First we'll set up and run the actual integration. Plotting and analysis will
	# come afterwards.

	x = np.array([1.0, 0.0, 0.0, 0.3])
	ti, tf = 0.0, 100.0
	hs = np.array([1.0, 0.1, 0.01])

	plt.figure('Problem 3: Part A', figsize=(10,10))
	subplot = 0 #211

	for h in hs:
		ts = myint.get_timesteps(ti, tf, h)
		
		res_e = myint.runner(myint.euler_stepper, f_orbit, x, ts, h)
		res_m = myint.runner(myint.mid_stepper, f_orbit, x, ts, h)
		res_r = myint.runner(myint.rk_stepper, f_orbit, x, ts, h)
		res_l = myint.leap_runner(f_orbit, x, ts, h)

		plt.subplot(231 + subplot).scatter(res_r[:,0], res_r[:,1], c='r', alpha=0.6)
		plt.subplot(231 + subplot).plot(res_r[:,0], res_r[:,1], c='k', alpha=0.6)
		plt.subplot(231 + subplot+3).scatter(res_l[:,0], res_l[:,1], c='r', alpha=0.6)
		plt.subplot(231 + subplot+3).plot(res_l[:,0], res_l[:,1], c='k', alpha=0.6)
		# plt.subplot(231 + subplot).set_xticklabels([])
		# plt.subplot(231 + subplot).set_yticklabels([])
		# plt.subplot(231 + subplot+3).set_xticklabels([])
		# plt.subplot(231 + subplot+3).set_yticklabels([])
		subplot += 1



	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()

def b():
	x = np.array([1.0, 0.0, 0.0, 0.3])
	ti, tf = 0.0, 100.0
	h = 0.1

	ts = myint.get_timesteps(ti, tf, h)
	res = myint.runner(myint.rk_stepper, f_orbit, x, ts, h)

	mid = len(res[0])/2 # m is for mid index of state vector
	# e =  (np.sum(np.power(res[:,m:],2), axis=1)/2.0) + (-1.0/np.power(1+np.sum(np.power(res[:,:m],2),axis=1),0.5))
	term1 = (np.sum(np.power(res[:,mid:],2), axis=1)/2.0)
	term2 = (-1.0/np.power(1+np.sum(np.power(res[:,:mid],2),axis=1),0.5))
	e = term1 + term2


	plt.scatter(ts, e, alpha=0.5,c='g')
	plt.scatter(ts, term1, c='r')
	plt.scatter(ts, term2, c='b')
	# plt.plot(ts,e,c='r')
	plt.xlim(ti-h,tf+h)
	# plt.ylim(-1,1)
	plt.show()




def main():
	print 'main'
	a()
	b()

if __name__ == '__main__':
	main()