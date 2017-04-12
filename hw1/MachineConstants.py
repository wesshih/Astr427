import numpy as np

'''
Wesley Shih
1237017
Astr 427 Homework 1
4/11/17

Problem 1: Machine Constants

The first problem of the homework asks us to empirically determine several
machine constants related to floating point numbers. These include the smallest number
epsilon that can be successfully added to or subracted from 1.0, and the maximum and minimum
positive numbers that can be represented using the float data type. We will be using 64-bit floats
for all parts of this problem.

This file contains code that calculates the machine constants and prints them to the console.
Becasue these values are fairly easy to calculate, this file will not contain any user-defined
functions or data structures.  For each part I will simply calculate the constant, print it to
the console, and comment on how it relates to the IEEE 754 representation.

A quick note on float representation that is relative to the whole problem.  For a 64-bit
float, there is 1 bit for sign, 11 bits for the exponent, and 52 bits for the Significand or fraction.
However, there is an implied 1 at the beginning of the significand, so we effectively have 53 bits
available for the fraction.
'''


# Part A
# We are looking for the smallest value that can be successfully subtracted from 1.0
# or formally, find smallest epsilon such that 1.0 - epsilon != 1.0
epsilon_a = 1.0
while (1.0 - epsilon_a/2.0) != 1.0:
	epsilon_a /= 2.0
print 'A) smallest epsilon s.t. 1.0 - epsilon != 1.0'
print '\t\tepsilon_a:\t' + `epsilon_a` + '\n'

# Running this code gives us a value of epsilon_a = 1.1102230246251565e-16
# This value is within an order of 2 of the true value of epsilon, as we know that
# 1.0 - (epsilon_a/2) == 1.0.  Given the 53 bits for the significand, we expect 
# the true machine epsilon to be 2^-(53 - 1). However, 2^-52 = 2.22e-16 which is basically
# double the value found above.




# Part B
# We are looking for the smallest value that can be successfully added from 1.0
# or formally, find smallest epsilon such that 1.0 + epsilon != 1.0
epsilon_b = 1.0
while (1.0 + epsilon_b/2.0) != 1.0:
	epsilon_b /= 2.0
print 'B) smallest epsilon s.t. 1.0 + epsilon != 1.0'
print '\t\tepsilon_b:\t' + `epsilon_b` + '\n'

# Running this code gives us a value of epsilon_b = 2.220446049250313e-16
# This value agrees very nicely with the "expected" epsilon I calculated above.
# 2^-52 = 2.22e-16, which is very close to the calculated value.



# Part C
# We are looking for the maximum number that can be represented with a float
max_num = 1.0
while (max_num * 2.0) != np.inf:
    max_num *= 2.0
print 'C) maximum representable number'
print '\t\tmax_num:\t' + `max_num` + '\n'

# Running this code gives us a max_num = 8.98846567431158e+307
# We know that this value is at least within an order of magnitude of the true max_num
# because we know that max_num * 2.0 == infinity representation.
# We have 11 bits total for the exponent, however these bits follow twos-compliment.
# This means we only have 10 bits available for positive exponents. So the maximum
# positive exponent is 1023.  We find that 2^1023 = 8.9884e+307, which is exactly what
# we have found here.  the true maximum number will be greater than this though, as we
# can increase the significand to push the max_num higher.


# Part D
# We are looking for the minimum representable positive number
min_num = 1.0
while (min_num/2) > 0:
    min_num /= 2 
print 'D) minimum representable number'
print '\t\tmin_num:\t' + `min_num` + '\n'

# Running this code gives us a min_num = 5e-324
# Like with max_num, to find the minimum number, we will look at the 11 exponent bits.
# However, this time we are able to use the MSB, and so we can achieve an exponent of -1024
# 2^-1024 = 5.56e-309. Using the exponent alone is not enough to get 5e-324.  To do this,
# we must denormalize the float, changing the implied 1.f to a 0.f.  This will get us the
# rest of the way there to 5e-324.

