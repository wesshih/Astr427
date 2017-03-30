import numpy as np


# Empirically determine the "Machine Constants" for 32-bit float

# A. 1.0 - epsilon != 1.0
# General idea is to start with epsilon at 1.0 and continue to divide by two until 1.0 - epsilon = 1.0
# Be sure to cast to float32 each time
epsilon = np.float32(1.0)
while np.float32(1.0 - epsilon/2) <> np.float32(1.0):
  epsilon = np.float32(epsilon/2)

print 'when epsilon = ' + `epsilon` + ', 1.0 - epsilon = ' + `np.float32(1.0 - epsilon)`
print 'when epsilon = ' + `np.float32(epsilon/2)` + ', 1.0 - epsilon = ' + `np.float32(1.0 - epsilon/2)`
print 'So smallest value of epsilon is ' + `epsilon`
print ''


# B. 1.0 + epsilon != 1.0
# Same idea as part A, except this time we'll add
epsilon = np.float32(1.0)
while np.float32(1.0 + epsilon/2) <> np.float32(1.0):
  epsilon = np.float32(epsilon/2)

print 'when epsilon = ' + `epsilon` + ', 1.0 + epsilon = ' + `np.float32(1.0 + epsilon)`
print 'when epsilon = ' + `np.float32(epsilon/2)` + ', 1.0 + epsilon = ' + `np.float32(1.0 + epsilon/2)`
print 'So smallest value of epsilon is ' + `epsilon`
print ''

# C. Maximum representable number
# I assume he means the max positive number?
# or does he mean all 1's (except sign bit) gives infinity?
max_num = np.float32(1.0)
while np.float32(max_num * 2) != np.inf:
  max_num = np.float32(max_num*2)

print 'max_num: ' + `max_num`
print 'max_num * 2: ' + `np.float32(max_num*2)`
print ''


# D. Minimum positive number
# similar idea as in C, except we'll divide and check > 0
min_num = np.float32(1.0)
while np.float32(min_num/2) > 0:
  min_num = np.float32(min_num/2)

print 'min_num: ' + `min_num`
print 'min_num/2: ' + `np.float32(min_num/2)`
