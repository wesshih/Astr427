'''
Wesley Shih
1237017
Astr 427 Homework 5

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