'''
Wesley Shih
1237017
Astr 427 Homework 5

'''

'''
NOTE: Unfortunately I learned a lesson to commit to git frequently. I lost a fair amount
of "clean up" and "commenting" work earlier today (6/7/17) and was forced to quickly
restore my work to a working version. This left me with very little time to comment
and explain my work.  This is not meant to be an excuse, but more of an explanation.
Because of this, I've decided to put all my comments in one section that describes the
project as a whole, while the individual files are not quite at the level that i would
usually like.

this file looks at the results calculated by the non-gpu python method and plots them
against the gpu method. The best estimate of pi using the non-gpu method was
pi = 3.14157264 with about 1 billion estimation points.

The best estimate of pi using the gpu was pi = 3.14159727 with the number of points 
equal to about 17 billion.

The times to execute these estimates varried significantly. the non-gpu method took
about 600 seconds (10 min) on my computer, where the gpu method took about 27 seconds.
For the GPU this includes both the setup and calculation times.

Ultimately I'm a bit disappointed that I wasn't able to provide the full analysis that I
was hoping for, however I'm very happy that I was able to program the gpu as well as I was
able to.  the files multipi.cu and multipi2.cu are my experimentations with using multiple
gpus simultaneously to calculate pi. I ran into some syncronization problems when copying
back to the host.  These files are messy but I decided to leave them in my git to show that
I had done them.

Overall I really enjoyed this project, even if I wasn't able to finish it as I had hoped.
I hope it won't be too much of an issue that this code isn't as well documented or explained
as my previous work. 

Thanks for a great quarter!
'''

import numpy as np
import matplotlib.pyplot as plt

gpu_data = np.loadtxt('gpu_results.dat')
python_data = np.loadtxt('python_results.dat')
gpu_num = [g[0]*g[1] for g in gpu_data]

gpu_error = [np.abs(np.pi - g[2]) for g in gpu_data]
python_error = [np.abs(np.pi - p[1]) for p in python_data]

print len(gpu_data)
print len(python_data)

plt.semilogx(gpu_num, gpu_error, 'ro-')
plt.semilogx(python_data[:,0], python_error, 'bs-')
plt.show()