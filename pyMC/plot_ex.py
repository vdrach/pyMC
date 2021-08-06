import numpy as np
import core
import analytical
import random

# Data for plotting
tau = np.arange(0.0, 5.0, 0.01)
m =analytical.m(tau)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plot the function
plt.plot(tau,m, 'r')
plt.xlim([0, 5])
plt.show()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plot the function

tau = np.arange(0.5, 5, 0.01)
U =analytical.U(tau)
plt.plot(tau,U, 'r')
plt.xlim([1, 4])
plt.show()
tau = np.arange(1.5, 3, 0.01)
C =analytical.C(tau)
print(C)
plt.plot(tau,C, 'r')
plt.xlim([1, 4])
plt.show()
