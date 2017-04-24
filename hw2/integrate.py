import numpy as np
import matplotlib.pyplot as plt

def euler_stepper(y, f, h):
	# y_n+1 = y_n + h*f(y,h)
	return y + h*f(y,h)

def mid_stepper(y, f, h):
	k1 = h*f(y,h)
	k2 = h*f(y+k1/2, h/2)
	return y + k2

def rk_stepper(y, f, h):
	k1 = h*f(y,h)
	k2 = h*f(y+k1/2, h/2)
	k3 = h*f(y+k2/2, h/2)
	k4 = h*f(y+k3, h)
	return y + (k1/6.0) + (k2/3.0) + (k3/3.0) + (k4/6.0)

def runner(y0, ti, tf, h, stepper, f):
	num = int((tf-ti)/h)+1 # number of steps to go from ti to tf by h. includes endpoint
	ys = np.zeros(shape=(num, 2))
	ys[0] = y0
	for i in xrange(1,num):
		ys[i] = stepper(ys[i-1], f, h)
	return ys


# start by making sure it works with 1-d equations of motion
# lets say that we throw a ball up with v0 = 10 m/s
# and let gravity be -10 m/s^2
# so our initial state vector = (0, 10)

# test for eq of motion with constant accel of -10
def f_motion(y, t):
	dydt = np.array([0,0]) 
	dydt[0] = y[1]
	dydt[1] = -10
	return dydt

def f_cos(y, t):
	dydt = np.array([0,0]) 
	dydt[0] = y[1]
	dydt[1] = -y[0]
	return dydt

y = np.array([1.0, 0.0])
ti, tf = 0.0, 30.0
h = 0.3

f = f_cos

yes = runner(y, ti, tf, h, euler_stepper, f)
yms = runner(y, ti, tf, h, mid_stepper, f)
yrs = runner(y, ti, tf, h, rk_stepper, f)

#print yes
#print yms
#print yrs

ycos =  np.cos(np.arange(ti, tf, h))

plt.scatter(xrange(len(yes)), yes[:,0],c='b')
plt.scatter(xrange(len(yms)), yms[:,0],c='r')
plt.scatter(xrange(len(yrs)), yrs[:,0],c='g')
plt.scatter(xrange(len(ycos)), ycos, c='c')
plt.show()
