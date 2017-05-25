'''
Wesley Shih
1237017
Astr 427 Homework 4
5/24/17

This file uses my golden search algorithm to fit a galaxy rotation model
to observed data.  Specifically, we will find the characteristic radius r0
that minimizes the residual sum of squares between observed data and our 
model.
'''

import numpy as np
import optimize as opt
import scipy.constants as sci
import matplotlib.pyplot as plt

def main():
	# first load the observed data file
	dat = np.loadtxt('rot.dat')

	# define things needed for model and fit
	v_inf = 100.0 # assympotic velocity i.e. v at r = infinity

	# finds the model v at a given radius for some characteristic radius
	f_model = lambda r,r0: v_inf * (1.0 - np.power(sci.golden, (-1.0*r)/r0))
	
	# calculates the residual sum of squares for the model with given r0
	# over the data points in dat
	f_rss = lambda r0: np.sum((dat[:,1] - f_model(dat[:,0], r0))**2)

	# Now that we are all set up, let's actually find the optimal r0.
	# In order to use golden search we need to establish a bracket (a,b,c)
	# such that f_rss(b) < f_rss(a), f_rss(c).  To find this bracket, plot
	# f_rss for a range of r0, and visually pick good values. We can't start
	# r0 range at 0.0 (due to div by 0), but try and get close

	for r0 in np.linspace(0.1, 10.0, 50):
		plt.scatter(r0, f_rss(r0))
	plt.title('RSS vs r0')
	plt.show()

	# Looking at this, we can establish an initial bracket. Clearly the minimum
	# of the function lies somewhere around r0 = 2.0, so let's select initial
	# bracket of a = 0.1, b = 4.0, and c = 10.0. Note that although our initial
	# (a,b,c) isn't spaced optimally, the golden search will quickly converge
	# to brackets that have the correct spacing (NR pg 494).

	res = opt.golden(f_rss, 0.1, 4.0, 10.0)
	print ''
	print 'result of golden search\t',res
	print 'r0 that best fits data\t',res[0]
	print 'rss of model with r0\t',res[1]

	# This tells us that using r0 = 1.60260821132 best fits the model
	# to the observed data. To confirm that this r0 is the model of best-fit,
	# let's plot it alongside the observed data. Also plot some of the
	# non-optimal models to visually confirm they fit the data worse.

	# plot non-optimal solutions
	for r0 in np.linspace(0.1, 10.0, 10):
		plt.plot(dat[:,0], f_model(dat[:,0],r0),'go-', alpha=0.5)

	# plot the optimal solution and observed data
	plt.plot(dat[:,0], f_model(dat[:,0], res[0]), 'bo-')
	plt.plot(dat[:,0], dat[:,1],'ro-')
	plt.title('Velocity vs Radius')
	plt.show()

	# Looking at this plot, we see that the model using the r0 found by
	# golden search (in blue) does closely follow the observed data (in red).
	# None of the non-optimal solutions (in green) appear to better fit the
	# observed data. Although only a small number of sub-optimal models are
	# shown, because we have actually optimized r0, we can be sure that the
	# blue model is actually the best fitting model.

if __name__ == '__main__':
	main()
