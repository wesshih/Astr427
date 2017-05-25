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


NUM_POINTS = 10000000 # number of points to generate

print 'beginning calc with NUM_POINTS = ', NUM_POINTS

rs = np.array([[r.random(), r.random()] for i in xrange(NUM_POINTS)])

start = time.time()
squared = np.power(rs,2)
summed = np.array([np.sum(s) for s in squared])
res = np.array([s for s in summed if s <= 1.0])
end = time.time()

print 'took', end-start,'seconds to calc first way'

start = time.time()
res2 = np.array([rr for rr in rs if np.sum(np.power(rr,2)) <= 1.0])
end = time.time()
print 'took', end-start,'seconds to calc second way'


print 'len rs\t',len(rs)
print 'len res\t',len(res2)
print 'area:', (4.0 * len(res2))/len(rs)

print 'len res\t',len(res)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,10))
# plt.scatter(res2[:,0], res2[:,1])
# plt.show()