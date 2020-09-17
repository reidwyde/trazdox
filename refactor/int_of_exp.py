import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np



eps = 1e-2
dose_time = np.array([[1],[0]])
dose = 10
def fun(t, y, eps):
    return -y + np.sum(np.abs(t-dose_time) < eps, axis=1)*dose/(2*eps)
t_span = (0, 10)
y0 = np.array([0,0])
ret = solve_ivp(fun, t_span, y0, method='Radau', dense_output=True, max_step=eps, args=(eps,))

ys = ret['y'][0]
ts = ret['t']

diffs = [t - ts[i - 1] for i, t in enumerate(ts)][1:]
diffs = [0] + diffs
diffs = np.array(diffs)

print(diffs)
print(diffs.shape)


ys = diffs*ys
accs = np.cumsum(ys)

plt.figure()
plt.plot(ret['t'], ret['y'][0])
plt.plot(ret['t'], accs)
plt.legend(["first order","second order"])
plt.savefig("first_and_second_order.jpg")
plt.close()


