'''
Wesley Shih
1237017
Astr 427 Homework 5
5/

This file is question 1 of homework 5, where we are calculating the value of pi
using the Monte-Carlo method NOT using gpus.
'''

import numpy as np
import time
import random as r

NUM_POINTS = 1000000000 # number of points to generate

print 'beginning calc with NUM_POINTS = ', NUM_POINTS

'''
Keep track of what was slow
for i in xrange(NUM_POINTS) was painfully slow. like 4s for 100,000

rs2 = np.array([sum([r.random(), r.random()]) for i in xrange(NUM_POINTS)])
	-> slower than just doing mag

l2 = sum([1 if m <= 1.0 else 0 for m in mag])
	-> np.where is much faster

xs = np.array([r.random()**2 for i in xrange(NUM_POINTS)])
ys = np.array([r.random()**2 for i in xrange(NUM_POINTS)])
mag2 = xs + ys
	-> just a bit slower than doing mag all in one go
'''


t_begin = time.time()
mag = np.array([r.random()**2 + r.random()**2 for i in xrange(NUM_POINTS)])
t1 = time.time()
print 'time to calc mags:',t1-t_begin


t1 = time.time()
l1 = len(np.where(mag <= 1.0)[0])
t2 = time.time()
print 'time to accept',t2-t1
print '\tl1',l1

print 'pi:',4.0*l1/NUM_POINTS

t_end = time.time()

print 'Total time for execution:', t_end-t_begin

def calc_pi(N):
	print 'calculating pi with',N,'samples'
