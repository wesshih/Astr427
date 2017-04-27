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

This file contains the f_orbit function which will be passed as a functor to the integrator.
The code for parts A and B are in the main function, and they are discussed more below.

As described in the integrate.py docstring, the following formats must be followed:

x = [x1, ... , xn, v1, ... , vn]
f has signature f(x, t)

'''
import numpy as np
import matplotlib.pyplot as plt
import integrate as myint

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
	mid = len(x)/2
	return np.concatenate((x[mid:], -x[:mid]/np.power(1.0 + np.sum(np.power(x[:mid], 2)), 1.5)))


def main():
	'''
	Problem 3 asks us to integrate the ode that describes orbital motion.
	This is very similar in its process to problem 2.  Like in problem 2,
	I will solve the equations, and then create the plots for parts A and B
	at the same time. Answers to questions and analysis of the problem is below.
	'''

	x0 = np.array([1.0, 0.0, 0.0, 0.3])
	ti, tf = 0.0, 100.0
	hs = np.linspace(1.0, 0.01, 4)

	# arrays for the energies of different solutions
	es_rk = np.array([])
	es_lf = np.array([])

	subplot = 221
	for h in hs:
		# Get a list of times ts that our integrator will use.  This list begins at the
		# initial time ti, and takes steps of size h until the final time tf. Then actually
		# run the integrator for the current stepper to get a solution.
		ts = myint.get_timesteps(ti, tf, h)
		sol_rk = myint.runner(myint.rk_stepper, f_orbit, x0, ts, h)
		sol_lf = myint.runner(myint.leap_stepper, f_orbit, x0, ts, h)

		# Plot things for Part A
		fig1 = plt.figure('Problem 3 - Part A: rk_stepper', figsize=(10,10))
		fig1.suptitle('Problem 3 - Part A: rk_stepper', fontsize=20)
		plt.subplot(subplot).plot(sol_rk[:,0], sol_rk[:,1], 'ro-', label=str('h: %0.3f'%h))
		plt.legend(fontsize=12)

		fig2 = plt.figure('Problem 3 - Part A: leap_stepper', figsize=(10,10))
		fig2.suptitle('Problem 3 - Part A: leap_stepper', fontsize=20)
		plt.subplot(subplot).plot(sol_lf[:,0], sol_lf[:,1], 'go-', label=str('h: %0.3f'%h))
		plt.legend(fontsize=12)
		

		# calculate the energy of the RK4 solution for part B
		fig3 = plt.figure('Problem 3 - Part B: Energy RK4', figsize=(10,10))
		fig3.suptitle('Problem 3 - Part B: Energy RK4', fontsize=20)
		mid = len(sol_rk[0])/2
		ke_rk = (np.sum(np.power(sol_rk[:,mid:],2), axis=1)/2.0)
		pe_rk = -1.0/(np.power(1 + np.sum(np.power(sol_rk[:,:mid],2),axis=1), 0.5))
		plt.subplot(subplot).plot(ts, ke_rk + pe_rk, 'ro-', label=str('h: %0.3f'%h))
		plt.subplot(subplot).plot(ts, ke_rk, 'go-', label=str('h: %0.3f'%h))
		plt.subplot(subplot).plot(ts, pe_rk, 'bo-', label=str('h: %0.3f'%h))
		plt.legend(fontsize=12)

		# now do the same for the leapfrog method.
		ke_lf = np.sum(np.power(sol_lf[:,mid:],2), axis=1)/2.0
		pe_lf = -1.0/np.power(1 + np.sum(np.power(sol_lf[:,:mid],2),axis=1), 0.5)
		fig4 = plt.figure('Problem 3 - Part B: Energy Leapfrog', figsize=(10,10))
		fig4.suptitle('Problem 3 - Part B: Energy Leapfrog', fontsize=20)
		plt.subplot(subplot).plot(ts, ke_lf + pe_lf, 'ro-', label=str('h: %0.3f'%h))
		plt.subplot(subplot).plot(ts, ke_lf, 'go-', label=str('h: %0.3f'%h))
		plt.subplot(subplot).plot(ts, pe_lf, 'bo-', label=str('h: %0.3f'%h))
		plt.legend(fontsize=12)

		subplot += 1


	plt.show()

''' Answers and Discussion

Part A)
		When we plot the solution to the orbital ODE found by RK4 or Leapfrog, we see what 
	appears to be an ellipse that is precessing around the origin.  This happens because the
	potential is not exactly 1/r. However, it is close enough to be mostly elliptical in shape.
	When comparing the RK4 and Leapfrog graphs we see that both solutions are very similar.
	At small step size h, the two are very comparable.  However, at larger h the Leapfrog
	graph appears a bit more symmetric than the RK4 graph.  This is a bit surprising, as I
	would have expected the 4th order method to produce more accurate estimation.

Part B)
		In the second part of problem 3, I calculated the energy E for both the RK4 and 
	Leapfrog solutions.  Beginning with the RK4 energies, we see that the total energy E (red)
	remains constant over the time interval.  This makes sense, as we would expect and hope
	that total energy is conserved.  After a closer look at the equation for E, I realized
	that the first term is simply the kinetic energy.  I split E into the kinetic and potential
	parts, and plotted them alongside E.  As we would expect in an orbiting scenario, the kinetic
	and potential energies oscillate back and forth, as one form of energy is converted into the other.
		The plots of Leapfrog show that unless step size is very small, the total energy estimated by
	the Leapfrog method is not constant.  I believe that this issue is very similar to the phase
	problems observed in Problem 2.  Because we are keeping track of positions and velocities at different
	times, it makes sense that the total energy can oscillate.  Depending on where we are in the orbit,
	we may be over or undercounting the total energy.  I believe that this problem can be solved by
	finding the velocities at the same times we know the positions, however I have not had time to test
	this myself.

'''

if __name__ == '__main__':
	main()