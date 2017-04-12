import numpy as np

'''
Wesley Shih
1237017
Astr 427 Homework 1
4/11/17

Problem 2: Roundoff Error
This file implenets a bit of code to investigate how some common operations behave for 
inputs on the order of 10^-7.  Specifically, we are interested the cosine and the square
of the input.  

As we saw in Problem 1, the smallest number we can add to a float and obtain a different float
is on the order of 10^-16.  This means that for numbers smaller than 10^-16 (which we can represent 
using a 64-bit float), function output may behave strangely.
'''


# this is the given function 1-cos(x)/x^2
def function(x):
	return (1.0-np.cos(x))/(x**2)

# starting with x = 10^-5, we will pass x to our function and save the output in an array
# I expect that numbers will start becoming strange around 10^-8 or -9.  I start at 10^-5
# to ensure that I have a decent set of points where the output is behaving "as expected"

x = 10.0**(-5)
xs, ys = [], []

# use inputs that span roughly 10^-5 to 10^-11. This should be more than enough to see the
# behavior we are looking for
for i in xrange(20):
	xs.append(x)
	ys.append(function(x))
	x /= 2.0


for i in xrange(20):
	print('x: %.3E\ty: %.3E\t1-cos[x]: %.3E' % (xs[i], ys[i],1-np.cos(xs[i])))

# The output shows a sudden change when x goes from 10^-8 to 10^-9.  To understand why
# this is happening, we should recall from problem 1 that the machine epsilon is on the
# order of 10^-16.  The main source of our problem is the cosine function. We see that 
# 1 - cos(x) goes to 0 when x is roughly 10^-9.  To see why this is happening, let's write
# the first few terms of cosine in series form
#
#	cos(x) = 1 - (1/2)(x^2) + (1/4!)(x^4) - (1/6!)(x^6) + ...
#
# If x is on the order of 10^-9, then x^2 is order 10^-18.  This number is below the machine
# epsilon, so 1 - (1/2)(10^-18) == 1.  Of course all higher order terms will fail as well, as
# they will be even smaller than the x^2 term.  So, for x < 10^-9, cos(x) = 1.
# This explains the sudden shift we see around x = 10^-9, however, we can see this effect earlier
# as well. The analytic limit as x -> 0 is equal to 1/2.  We see that our function agrees pretty
# well with this around x~10^-5.  As we decrease x, higher order cosine terms stop making
# corrections, which is evident by the function output wandering slightly when x~10^-7 or -8.
