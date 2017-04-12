import numpy as np
# import matplotlib.pyplot as plt

'''
Wesley Shih
1237017
Astr 427 Homework 1
4/11/17

Problem 3: Interpolation

This problem asks us to implement a program that can read in a data file and interpolate
between the given values.  We begin by implementing a linear interpolator, and then move
on to using Neville's algorithm to fit a polynomial to the given data and use that as our
interpolating function.

Although parts A and B are concerned with Linear Interpolation only, I have used my code from
part C to find a Linear fit.  If we think of the linear interpolation as being a 1st order
polynomial fit, then as long as we can specify the desired order to neville's algorithm, we can
use it to find 1st order.

This file contains three functions: neville, neville_order, and main.  The main function is where
the execution starts, and where the I provide answers to parts A, B, and D. 
'''

def neville(xs, ys, x):
    """
    Run Neville's algorithm on a given xs,ys to approximate f at x.

    this method will run neville's algorithm on the given lists which represent the known
    x and y values.  This method will use all given points in the approximation.  The order of the 
    approx can be specified by passing neville lists with order + 1 elements in it. x must lie within
    the given range of x values.

    Args:
    xs (list):  list of floats for x values of known points. Constant spacing is expected
    ys (list):  list of floats for y values of known points.
    x  (float): The x value of where we want to approximate the function.

    Returns:
    float:    function returns a float that is the approximation of function at x.

    """
    ps = np.copy(ys).tolist()
    n = len(xs)
    ps_offset = 0
    for i in xrange(1, n):
        for j in xrange(n-i):
            p = ((x - xs[j+i])*ps[j+ps_offset] + (xs[j] - x)*ps[j+ps_offset+1])/(xs[j] - xs[j+i])
            ps.append(p)
        ps_offset += n-i+1
    return ps[-1]



def neville_order(xs, ys, x, order):
    """
    Uses neville's algorithm to fit a polynomial of given "order" to approximate the function at x

    Args:
    xs  (list)    list of floats for x values of known points
    ys  (list)    list of floats for y values of known points
    x   (float)   the x value of the point we wish to approximate
    order (int)   specifies the desired order of the fit. 1: linear, 2: quadratic, 3: cubic, etc.

    Returns:
    float:  Returns the approximated value of the function at the specified x.
    """
    if x in xs: # no need to interpolate if x is at a known spot
        return ys[np.where(x == xs)[0][0]]

    if not order < len(xs):
        raise Exception('neville_order: order > len(xs). Cannot calculate this')

    n = len(xs)
    a = (int) (n * (x-xs[0])/(xs[-1]-xs[0]))  # index of closest known point
    b = a + 1 if xs[a] < x else a - 1
    if a > b: a,b = b,a
    assert(a >= 0)
    assert(b < n)

    # now we need to widen the range of [a,b] if order > 1.  If this expansion is
    # asymetric, then we should attempt to first go in direction of index closest to x.
    # these are the amounts we need to expand on either side (a big and a small if asymetric)
    exp_sm = (order-1)/2
    exp_bg = order-1-exp_sm
    if xs[b] - x > x - xs[a]: # is x closer to xs[a] or xs[b]?
        a -= exp_bg
        b += exp_sm
    else:
        a -= exp_sm
        b += exp_bg

    if a < 0:
        b = min(b-a, n-1)
        a = 0
    if b > n-1:
        a = max(a+(n-1-b),0)
        b = n -1

    return neville(xs[a:b+1],ys[a:b+1],x)

def main():
    
    # Parts A and B
    # The objective of parts A was to write a program that could read a set
    # of values from a file, and then perform linearlly interpolation at an 
    # arbitrary point.  Part B was to use this program on hw1.dat at x = 0.75.
    # Because my neville's algorith is fairly general and can calculate different
    # order polynomials, I can just use it to find the 1st order (linear) polynomial.
    data = np.loadtxt('hw1.dat')
    xs = data[:,0]
    ys = data[:,1]
    linres = neville_order(xs, ys, 0.75, 1)
    print 'Linear Interpolation of hw1.dat at x = 0.75 gives y = ' + `linres`

    # Part C
    # Here I use neville's algorithm to find a 4th order polynomial for the data
    # in hw1.dat.  I then use this polynomial to estimate the valut at x = 0.75.
    polyres = neville_order(xs, ys, 0.75, 4)
    print '4th order polynomial of hw1.dat at x = 0.75 gives y = ' + `polyres`

    # Part D
    # The actual function is y = 1/(1 + 25x^2). So when x = 0.75, y = 0.06639.
    # our linear estimation of y = 0.0882, and our 4th order estimation gives us
    # y = -0.356.  Our 4th order polynomial gives us a "worse" estimation because 
    # it must go through our 5 known points in a smooth fashion.  For the points at
    # x = -0.5, 0, 0.5, our 4th order poly will be concave down.  So, on the intervals
    # (-1.0, -0.5) and (0.5, 1.0), the 4th order polynomial will be concave up.  This means
    # that for any x in these ranges, we will be estimating a value less than the actual value
    # of the function.  So for this particual case, we have created an estimation that
    # essentially overfits the given data.  For this particular function, we would be better
    # off using a lower order approximation.

    # The following code was used to graph the polynomials of various order
    # I decided to leave this part in because it is interesting to see the polynomials
    # plotted against one another.  If you wish to uncomment and run this code, be sure
    # to uncomment the matplotlib import at the top of the file

    # test_x = np.arange(-1.0, 1.0, 0.05).tolist()
    # test_y1 = []
    # test_y2 = []
    # test_y3 = []
    # test_y4 = []
    # for x in test_x:
    #     test_y1.append(neville_order(xs, ys, x, 1))
    #     test_y2.append(neville_order(xs, ys, x, 2))
    #     test_y3.append(neville_order(xs, ys, x, 3))
    #     test_y4.append(neville_order(xs, ys, x, 4))
    # plt.scatter(test_x, test_y1, c='b')
    # plt.scatter(test_x, test_y2, c='r')
    # plt.scatter(test_x, test_y3, c='g')
    # plt.scatter(test_x, test_y4, c='c')
    # plt.plot(test_x, test_y1, c='b')
    # plt.plot(test_x, test_y2, c='r')
    # plt.plot(test_x, test_y3, c='g')
    # plt.plot(test_x, test_y4, c='c')
    # plt.show()
  
if __name__ == '__main__':
    main()