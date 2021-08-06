import numpy as np
import core
import analytical
import random

import numba
from timeit import default_timer as timer

import matplotlib.pyplot as plt

N = 20
Niter = 1200*int(1e3) # in MCS
Ntherm = 10000
J = 1
B = 0
print(f"Niter={Niter}, Lattice size = {N}x{N}")

tau_range=np.arange(0.2,3,0.1)

MCruns = core.DataBaseMC()
meas_freq = 10

for tau in tau_range:
    beta = 1/(tau*J)
    print(f"beta={beta}, J={J}, B={B}, tau={tau}")
    conf = core.Ising(shape=(N,N),BC='periodic',flag_init_conf="random",beta=beta,J=J,B=B) 

    H = conf.H()
   
    sim = core.MC(conf,Niter,obs=None, measure_freq=meas_freq)
    db_el = sim.run()
    MCruns.append(db_el)

#average over ensemble for each tau
resE = []
resE2 =[]
resmag =[]
resC = []
therm = 2000

for i in range(len(tau_range)):
    beta = 1/(tau_range[i]*J)
    resE.append( np.mean(MCruns.db[i].dat['H'][therm:]))
    resE2.append( np.mean(MCruns.db[i].dat['H2'][therm:]))
    resmag.append( np.abs(np.mean(MCruns.db[i].dat['m'][therm:])))
    resC.append( (np.mean(MCruns.db[i].dat['H2'][therm:]) - np.mean(MCruns.db[i].dat['H'][therm:])**2)*beta**2 )
    


plt.plot(tau_range,resmag)
plt.ylabel('magnetization')
plt.show()


plt.plot(tau_range,resC)
plt.ylabel('Specific heat')
plt.show()
