import numpy as np
from time import time
  
  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


class MC:
    def __init__(self,field,Niter,obs, measure_freq):
        self.field = field
        self.Niter = Niter
        self.obs = obs
        self.measure_freq= measure_freq
        self.H = []
        self.H2 = []
        self.m = []
    @timer_func
    def run(self):
        H= self.field.H()
        for i in range(self.Niter):
            DH = self.field.update()
            H += DH
            if i%self.measure_freq==0:
                self.H.append(H)
                self.H2.append(H**2)
                self.m.append(self.field.mag())
        
        datMC = DataMC(self.field.J,self.field.B,self.field.beta,self.field.lat.shape,self.Niter,self.measure_freq,{'H':self.H,'H2':self.H2,'m':self.m})
        return datMC




class DataMC:
    def __init__(self,J,B,beta,shape,Niter,measure_freq,dic):
        self.J = J
        self.B = B
        self.beta = beta
        self.shape = shape
        self.Niter = Niter
        self.dat = dic
        self.measure_freq= measure_freq
    def summary(self,therm):
        print(f"Total number of MC steps: {self.Niter}, measurements frequency {self.measure_freq}")
        print(f"Thermalisation skip first {therm} measurements")

        for key,value in self.dat.items():
            print(f"obs: {key}, av = {np.mean(value[therm:])} +/- {np.std(value[therm:])}")

    def get_obs(self,k):
        return self.dat[k]

class DataBaseMC:
    def __init__(self):
        self.db = []
        self.n = len(self.db)

    def append(self,datMC):
        self.db.append(datMC)
        self.n +=1

    def get_info(self):
        print(f"Number of simulations stored {len(db)}")
    
    def store(self):
        pass
