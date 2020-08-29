import matplotlib.pyplot as plt

#import pymc3

from scipy.integrate import solve_ivp
import numpy as np



eps = 1e-2
#impulses must be separated by at least 2*eps in the time domain
dose_time = np.array([[5, 5+2*eps],[2, 7]])
dose = 10
def fun(t, y, eps):
    return -y + np.sum(np.abs(t-dose_time) < eps, axis=1)*dose/(2*eps)
t_span = (0, 10)
y0 = np.array([20,10])
ret = solve_ivp(fun, t_span, y0, method='Radau', dense_output=True, max_step=eps, args=(eps,))















plt.figure()
plt.plot(ret['t'], ret['y'][0])
plt.plot(ret['t'], ret['y'][1])
plt.legend(["0","1"])
plt.savefig("times.jpg")
plt.close()


