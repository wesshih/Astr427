'''
Wesley Shih
1237017
Astr 427 Homework 2
4/25/17

Problem 2: Diff Eq for cosine

In this problem we are asked to solve the differential equation of d^2x/dt^2 + x = 0.
We know this has an analytic solution of x(t) = cos(t), however we will use the functions
from problem 1 to solve this numerically.

This file contains a function f_cos which will act as the functor to solve this Diff Eq.
The rest of the problem is addressed in the main function below.

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
	return np.concatenate((x[len(x)/2:], -x[:len(x)/2]))


def main():
	'''
	In problem 2 we are first asked to solve the given differential equation using a variety
	of methods and step sizes.  Each integration method and step size will give us a set of
	points that approximate the actual solution with varying levels of success.  By plotting
	them against the known analytic solution of cos(t), we can see how well the various
	methods did.  Part B asks us to look at the error of our solution with respect to step size.

	Because both parts require us to solve the same Diff Eq, we can avoid redundancy in the code
	by only solving these equations one time.  I will do the calculations for both parts at the same
	time, and then return to analysis and questions at the end.  This means that some code for part
	B is mixed in with code for part A, however, I think that it is worth it to only have to run the
	solver one time.
	'''

	# Create the initial state vector array, define the initial and final time, and
	# create array for the various step sizes we will test
	x0 = np.array([1.0, 0.0])
	ti, tf = 0.0, 30.0
	hs = np.array([1.0, 0.3, 0.1, 0.03, 0.01])

	# These values will be used to calculate and plot the error in part B
	cos30 = np.cos(30.0)
	fig2 = plt.figure('Part B: Error', figsize=(10,10))
	fig2.suptitle("Part B: Error", fontsize=25)
	labels_B = []

	colors = iter(['r', 'b', 'g', 'c', 'm'])

	# Use each of the steppers to solve the ODE for the various step sizes
	steppers = [myint.euler_stepper, myint.mid_stepper, myint.rk_stepper, myint.leap_stepper]
	for step in steppers:

		# To view how the solution changes with step size, a set of plots for each stepper
		# algorithm will be created.  Each stepper will have its own plot window with 5 subplots
		# that show how the estimation varies with step size.
		fig = plt.figure('Part A - Stepper: ' + `step.__name__`, figsize=(10,10))
		fig.suptitle(step.__name__, fontsize=25)
		subplot = 511

		# array used to store error terms for part B
		err = np.array([])

		for h in hs:
			# Get a list of times ts that our integrator will use.  This list begins at the
			# initial time ti, and takes steps of size h until the final time tf. Then actually
			# run the integrator for the current stepper to get a solution.
			ts = myint.get_timesteps(ti, tf, h)
			sol = myint.runner(step, f_cos, x0, ts, h)
			err = np.append(err, np.abs(sol[-1,0] - cos30))

			# Graph the solution for part A in a format that will hopefully be helpful
			plt.subplot(subplot).plot(ts, sol[:,0], 'o-', c='r', label=str('h: %03f'%h), lw=5.0)
			plt.subplot(subplot).plot(ts, np.cos(ts), '-' ,c='k', label='cos(t)', lw=2.0)
			plt.subplot(subplot).set_xlim(-1, 31)
			plt.subplot(subplot).set_ylim(-3,3)
			if (step.__name__ == 'euler_stepper' and h > 0.1):
				# need to change bounds because the euler solutions quickly goes off the screen
				plt.subplot(subplot).set_ylim(-15,15)
			plt.subplot(subplot).set_axis_bgcolor('lightgray')
			plt.legend(ncol=2, loc=3,fontsize=12)
			subplot += 1

		# We now have the error terms for some stepper function over all step step sizes.  We can
		# add these to the plot for part B, which will compare the various errors.
		plt.figure(fig2.number)
		labels_B.append((plt.loglog(hs, err, 'o-', c=colors.next(), lw=2.5)[0],step.__name__))

	# Add legends and labels to Part B plots, and then show all plots we have made
	plt.figure(fig2.number)
	labels_B = zip(*labels_B)
	fig2.legend(labels_B[0], labels_B[1], 'lower center', ncol=2)
	plt.show()


''' Analysis and Discussion

Part A)
		It wasn't until the analysis stage of the homework that I realized that I didn't have to 
	implement the midpoint method.  However, I think it can still be useful in my analysis, so
	I'll leave it in.
		The calculated solutions are plotted in red alongside the analytic solution which is
	in black.  When the step size is large it is easy to tell the difference between these two lines,
	however as the step size decreases, the black borders of the plotted points overlap and effectively
	make the entire plot look black.  For the most part this is okay, because the calculated solutions
	tend to more closely follow the analytic solution as we decrease step size.
		Starting with the methods that estimate the solution very accurately, the RK4 technique
	follows the true cosine function well, even with a large step size (h = 1.0).  This is because
	for every step we calculate 4 separate derivatives. This means that as step size decreases, the
	efficency will decrease quickly.  Overall, the RK4 method accurately estimates the cosine function
	for the given step sizes.
		The Leapfrog method is the next best at following the analytic solution.  When h < 0.3
	the leapfrog method does a good job of reproducing the analytic solution.  However, for larger
	step sizes, this method runs into phase issues.  This makes sense, because we are estimating
	positions and velocities at different times.  If the step size is large, then we are using a
	velocity that isn't very close in time to that of the position we are estimating (or vice versa).
	This offset between the position and velocity times is what is causing the phase shift at big h.
		On the other end of the spectrum is the euler method.  If the step size is decently small, then 
	any potential change in the state vector will also be small.  So, for h = 0.01, the euler solution
	follows cosine fairly decently.  However, as the step size gets larger, the euler method tends to
	snowball out of control.  For larger h, the euler approximation increases amplitude as time goes on.
	Once this starts, it is pretty much impossible to recover.  This effect is worse at bigger h.

Part B)
		The log-log plot shows how the difference between the calculated solution and the actual solution
	vary with step size.  The values we are comparing are at the end of the time interval, which we can 
	use to analyze how well a particular method would do over a large time scale. The plot shows that as
	we decrease step size, the error in our calculation decreases as well (as we would expect).
	We can compare the error of the various methods by looking at the slopes of each line.  The steeper
	the slope, faster the error will approach 0 as we decrease h.  The RK4 method does the best here,
	which is consistent with our observation that RK4 provided the best fit solutions in part A.  We can 
	estimate the slope of this line to be about 4 on the log-log scale.  This translates to something
	to the 4th power if we convert out of log-log space. This makes a lot of sense, as RK4 is an 
	estimator that is 4th order in h.
		The Leapfrog method increases from roughly 10^-4 to 10^0 as h goes from 10^-2 to 10^0.  This
	gives the Leapfrog a slope of roughly 2.  This makes sense given that the Leapfrog method is second
	order in h.  This is the same order as the midpoint method.  So, we would expect the slope of the 
	midpoint line should be roughly the same as Leapfrog at 2.  We see this on the log-log plot, which
	reinforces my thoughts that this is the "right" way to read this plot.
		Finally, we know that the euler method is first order in h, so we would expect the slope of 
	the log-log line to be 1.  However, we do not see this, as the euler line is all over the place.
	This is not unreasonable given that we saw very chaotic behavior from euler in part A.
'''

if __name__ == '__main__':
	main()