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

def calc_pi(N):
	print 'beginning calc with NUM_POINTS = ',N

	t_begin = time.time()
	mag = np.array([r.random()**2 + r.random()**2 for i in xrange(N)])
	t1 = time.time()
	print 'time to calc mags:',t1-t_begin

	t1 = time.time()
	l1 = len(np.where(mag <= 1.0)[0])
	t2 = time.time()
	print 'time to accept',t2-t1
	print '\tl1',l1

	pi = 4.0*l1/N
	print 'pi:',pi

	t_end = time.time()
	print 'Total time for execution:', t_end-t_begin
	return pi


pis = np.array([])
max_i = 20
for i in range(max_i):
	NUM_POINTS = 1024 * (1<<i)
	pis = np.append(pis, calc_pi(NUM_POINTS))

print 'pis:',pis
assert(len(pis)==max_i)
file = open('python_results.dat','w')
for i in range(max_i):
  file.write(str(1024*(1<<i)))
  file.write(' ')
  file.write(str(pis[i]))
  file.write('\n')

print 'done'