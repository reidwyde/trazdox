import matplotlib.pyplot as plt

#import pymc3

from scipy.integrate import solve_ivp
import numpy as np



#impulses must be separated by at least 2*eps in the time domain
#impulse can not be placed before 1 eps in the time domain
dose_time = np.array([[5, 0],[2, 7]])
dose = 50

def fun(t, y, dose_time=dose_time, eps=1e-2):
    return -y + np.sum(np.abs(t-dose_time) < eps, axis=1)*dose/(2*eps)


#domain to simulate over
t_span = (0, 10)

#initial conditions
y0 = np.array([0,0])

#ret is a dict containing the integration of the DE
ret = solve_ivp(fun, t_span, y0, method='Radau', dense_output=True, max_step=eps, args=(eps,))


plt.figure()
plt.plot(ret['t'], ret['y'][0])
plt.plot(ret['t'], ret['y'][1])
plt.legend(["0","1"])
plt.savefig("times.jpg")
plt.close()




