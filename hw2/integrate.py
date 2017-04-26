'''
Wesley Shih
1237017
Astr 427 Homework 2
4/25/17

Problem 1: ODE Integrator

This file contains functions that can be used to solve ODEs. 

There are a few "names/arrays/objects" that will be used in all parts of this homework.
The format and definition of these objects should be consistent across all hw2 files.
I will put a detailed description of these objects here, but will only restate the required 
format in subsequent comments.

x: Numpy array that holds information about the dependent variables at some time.  We can
	think of this as a vector that represents the state of the system at one particualr time,
	and so I will generally refer to it as a "state vector". I will try to keep the definition
	of x as general as possible for potential future use. If we have n dependent varialbes
	x1 to xn, then the required format for x is:
		-x must be of length 2n -> len(x) = 2n
		-The first n elements are the dependent variables x1 to xn -> x[:n] = [x1,...,xn]
		-The remaining n elements are the 1st time derivative of x1 to xn -> x[n:] = [v1,...,vn]
		-In short, x must be of form [x1,...,xn, v1,...,vn] for the integrator functions to work
	
	(A note to myself for later: there is no time dependence in this assignment, however, these
	x1 to xn should probably be functions that take a time t)


f: Function that differentiates a given state vector at the given time.  This will primarily
	be used as a functor, as it will be used by the integrator functions to determine how the 
	state vector changes with each step. This function will depend on the particular Diff Eq
	that we are tying to solve.  For f to be used in my integrator, it must have the function
	signature f(x, t), where x is a state vector and t is time.  f must return a properly
	formatted state vector (that has been differentiated).

stepper: There are several steppers in this file.  They all aim to do the same thing, but the
	implementation differs between them.  The job of the stepper is to use the given function f
	to step a state vector forward by h.  Although this assignment doesn't have time dependent
	equations, the stepper may need to know the beginning time of the step.  Any stepper must
	have the function signature stepper(f, x, t, h), where f is the function described above,
	x is the state vector at time t, and h is the step size.  Stepper should return a state
	vector that has been moved forwared in time by one step.

runner:	The runner function is the top level function that actually calculates the entire problem.
	Given correctly formatted stepper, f, x0, h, and initial and final times ti and tf, the runner
	integrates the ODE over the given time frame.  Beginning at ti, the runner uses the stepper
	to move x forward by h until it reaches tf.  When finished, the runner returns an array of
	state vectors, one for each time step (including ti and tf).

'''

import numpy as np
import operator


def runner(step, f, x, t, h):
	'''
	Solves an ODE by repeatedly calling stepper to time evolve the state vector x. Begins
	at time ti, and moves towards tf in steps of h.

	Args:
		step:	stepper algirthm to use
		f:		derivative function to use
		x:		Initial state vector
		t:		time interval to step through
		h:		Step size

	Returns:
		x: 		An array of state vectors that holds the result of the stepper at each time step
				from ti to tf, including the initial and final values.
	'''

	x = np.array([x])
	for ti in t[:-1]: # t[:-1] because t[-1] = tf, so we're done
		x = np.append(x, [step(f, x[-1], ti, h)], axis=0)
	return x

	'''
	This is a second runner that i am playing around with different ways to do it,
	This shouldn't be in the project when i submit...
	'''
	# x = np.append([x], np.zeros(shape=(len(t)-1, len(x))), axis=0)
	# for i in xrange(1,len(t)):
	# 	x[i] = np.array([step(f, x[i-1], t[i], h)])
	# # OR.....
	# # map(lambda i: operator.setitem(x, i, step(f, x[i-1], t[i-1],h)), xrange(1,len(x)))
	# return x


def euler_stepper(f, x, t, h):
	'''
	Uses the euler method to move the state vector x forward to time t + h.
	x(t+h) = x(t) + h * f(x, t)

	Args:
		f: Derivative function to be applied to state vector x
		x: state vector before step. formatted as described above
		t: beginning time of this step
		h: Step size

	Returns:
		the state vector after a step of h
	'''
	return x + h*f(x, t)


def mid_stepper(f, x, t, h):
	'''
	Uses the midpoint method to time evolve the state vector x. First determines
	the position and velocity halfway through the step, and then uses this value
	to get the state after the full step.

	Args:
		f: Derivative function to be applied to state vector x
		x: state vector before step. formatted as described above
		t: beginning time of this step
		h: Step size

	Return:
		state vector after a step of h
	'''
	k1 = h*f(x,t)
	k2 = h*f(x+k1/2.0, t+h/2.0)
	return x + k2


def rk_stepper(f, x, t, h):
	'''
	Uses the 4th order Runge-Kutta method to move the state vector forward by h.
	This method uses a weighted average of 4 different estimates of the state between
	t and t + h.

	Args:
		f: Derivative function to be applied to state vector x
		x: state vector before step. formatted as described above
		t: beginning time of this step
		h: Step size

	Return:
		state vector after a step of h
	'''
	k1 = h*f(x,t)
	k2 = h*f(x+k1/2.0, t+h/2.0)
	k3 = h*f(x+k2/2.0, t+h/2.0)
	k4 = h*f(x+k3, t)
	return x + (k1/6.0) + (k2/3.0) + (k3/3.0) + (k4/6.0)


def leap_runner2(f, x, ti, tf, h):
	'''
	Special runner that is uses the leapfrog stepper to perform the integration.  A seperate
	runner is needed for the leapfrog method becasue it requires a slightly different setup
	and finish.  We first find the velocity a one half step from ti.  We will use the rk_stepper
	to find these values.  It is okay to use an "expensive" method like RK4, as we will only run
	it once to get an initial value.  We will then continue forward using the leap_stepper, where 
	we will have positions x1 to xn on integer multipls of h, and v1 to vn on half integers.

	Args:
		f:		derivative function to use
		x:		Initial state vector
		ti:		Initial time
		tf: 	Final time
		h:		Step size

	Returns:
		x: 		An array of state vectors that holds the result of the stepper at each time step
				from ti to tf, including the initial and final values.
	'''
	mid = len(x)/2
	# first entry in array of state vectors is x[0] = [x1(ti), ... , xn(ti), v1(ti+h/2), ... , vn(ti+h/2)]
	x = np.array([np.append(x[:mid], rk_stepper(f, x, ti, h/2.0)[mid:])]) 
	while ti < tf:
		x = np.append(x, [leap_stepper(f, x[-1], ti, h)], axis=0)
		ti += h

	# we now have x(tf), but v(tf+h/2). so lets find v(tf).
	# is this step necessary? and if so, is it necessary atfer ever step?
	# i.e. should the state vector we return be [xi(ti),vi(ti)] or [xi(ti),vi(ti+h/2)]?
	# x[-1,mid:] = ((x[-1,:mid] - x[-2,:mid])/h)

	return x

# Question: can i incorporate this into the general runner? then i can iterate more easily
def leap_runner(f, x, t, h):
	mid = len(x)/2
	x = np.array([np.append(x[:mid], rk_stepper(f, x, t[0], h/2.0)[mid:])])
	for ti in t[:-1]: # again the -1 is because we want to stop stepping 1 from the end
		x = np.append(x, [leap_stepper(f, x[-1], ti, h)], axis=0)
	return x


def leap_stepper(f, x, t, h):
	'''
	Uses the leapfrog method to move the state vector forward by one step.  It is a bit
	unique in that the times we know x1 to xn are offset by h/2 from the times we know v1 to vn.
	Ultimately this allows us to reuse calculated values in such a way that we get a solution
	that is of order O(h^3), but only requries one derivative per step.  However, it is important
	to remember that we must convert our velocities to integer multiples of h if we need position
	and velocity at the same time.

	Args:
		f: Derivative function to be applied to state vector x
		x: state vector before step. formatted as described above
		t: beginning time of this step
		h: Step size

	Return:
		state vector after a step of h
	'''
	# x(i+1) = x(i) + h*v(i+1/2)
	# v(i+3/2) = v(i+1/2) + h*f(x(i+1), h)

	mid = len(x)/2
	x[:mid] = x[:mid] + h*x[mid:]
	x[mid:] = x[mid:] + h*f(x, t/2.0)[mid:]
	return x

def get_timesteps(ti, tf, h):
	return np.linspace(ti, tf, int((tf-ti)/h)+1)
