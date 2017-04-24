import numpy as np
import matplotlib as pylot

def euler_stepper(y, f, h):
	# y_n+1 = y_n + h*f(y,h)
	return y + h*f(y,h)

def mid_stepper(y, f, h):
	k1 = h*f(y,h)
	k2 = h*f(y+k1/2, h/2)
	return y + k2

#def rk_stepper(y, f, h):
#	print 'rk_stepper'
#
#def runner(y0, t0, tn, h):



# start by making sure it works with 1-d equations of motion
# lets say that we throw a ball up with v0 = 10 m/s
# and let gravity be -10 m/s^2
# so our initial state vector = (0, 10)

# test for eq of motion with constant accel of -10
def f(y, t):
	dydt = np.array([0,0]) 
	dydt[0] = y[1]
	dydt[1] = -10
	return dydt

# set up our initial "state" vector
y = np.array([0, 10]) #x0 = 0, v0 = 10
# lets find the eq over 2 seconds (1 up and 1 down)
ti, tf = 0.0, 2.0
# lets let the step size be 0.25 seconds (4 iterations each direction)
h = 0.25

num = int((tf-ti)/h)
print num
yes = np.zeros(shape=(num, 2)) #np.array([0,0]*num)
yms = np.zeros(shape=(num, 2)) #np.array([0,0]*num)

yes[0] = y
yms[0] = y

for i in xrange(1,num):
	yes[i] = euler_stepper(yes[i-1], f, h)
	yms[i] = mid_stepper(yms[i-1], f, h)

print yes
print yms

print yes[:,0]
