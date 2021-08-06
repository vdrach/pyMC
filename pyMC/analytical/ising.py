import numpy as np
from numba import jit
from scipy.integrate import quad
from scipy.misc import derivative

def m(tau):
    tau_c = 2/(np.log(1+np.sqrt(2)))
    return np.where(tau < tau_c, (1- np.sinh(2/tau)**(-4))**(1/8)  ,0)
    

def Capprox(tau):
    tau_c = 2/(np.log(1+np.sqrt(2)))
    return (2/tau)**2*(-np.log(1-tau/tau_c) - np.log(2*tau) - (1+np.pi/4))


def integrand(phi, kappa):
    return np.log(1+np.sqrt(1-kappa**2*np.sin(phi)**2))


# exact results
@jit
def logZ(tau):
    i = 0
    res = np.zeros(len(tau))
    for t in tau:
        betaJ = 1/t
        two_betaJ = 2*betaJ
        kappa = 2*np.sinh(two_betaJ)/(np.cosh(two_betaJ)**2)
        term1 = np.log(np.sqrt(2)*np.cosh(two_betaJ))
        term2,err = quad(integrand,0,np.pi/2,args=(kappa,))
        res[i] = term1 + term2/np.pi
        i+=1
    return  res

@jit
def U(tau): # internal energy in unit of J 
    return  tau**2*derivative(logZ,tau,dx=1e-6,n=1)

@jit
def C(tau): # this is C/k
    return tau**2*derivative(logZ,tau,dx=1e-6,n=2)  + 2*tau*derivative(logZ,tau,dx=1e-6,n=1)
