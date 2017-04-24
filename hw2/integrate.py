import numpy as np

def euler_stepper(y, f, h):
	print 'euler_stepper'
	# y_n+1 = y_n + h*f(y,h)
	return y + h*f(y,h)

def mid_stepper(y, f, h):
	print 'mid_stepper'
	# k1 = h*f(y,h)
	# k2 = h*f(y+k1/2, h/2)

def rk_stepper(y, f, h):
	print 'rk_stepper'

def f(y, h):
	print 'generic derivative f function'

def main():
	print 'main'

if __name__ == '__main__':
	main()
