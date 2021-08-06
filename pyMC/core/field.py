import numpy as np
from math import prod
import numba
from .lattice import *

class Field:

    def __init__(self,shape,BC,target_space,dtype):
        self.target_space = target_space
        self.dat = np.array(prod(shape),dtype=dtype)
        self.lat = Lattice(shape,BC='periodic')      

    def mem_use(self):
        print(f"Memory used by data is {self.dat.nbytes} bytes")
        print(f"Memory used by lat.neighbours is {self.lat.neighbours.nbytes} bytes")
    # add multiplication of fields
    
    

class Ising(Field):
    def __init__(self,shape,BC,flag_init_conf,beta,J,B):
        super().__init__(shape,BC,target_space=(-1,1),dtype=np.int64)
 
        if flag_init_conf == 'random':
            self.dat = np.random.choice(self.target_space,size=self.lat.V)
            self.dat = self.dat.astype(np.int64)

        if flag_init_conf == 'up':            
            self.dat = np.ones(self.lat.V,dtype=np.int64)
        if flag_init_conf == 'down':
            self.dat = -np.ones(self.lat.V,dtype=np.int64)
        self.J = J
        self.B = B
        self.beta = beta

    @property
    def J(self):
        return self.__J
    @property
    def B(self):
        return self.__B
    @property
    def beta(self):
        return self.__beta
    @B.setter
    def B(self,var):
        if type(var) == float or int:
            self.__B = var
        else:
            raise Exception("Ising Class: B must be real. Exception while setting J!")
    @beta.setter
    def beta(self,var):
        if var > 0.:
            self.__beta = var
        else:
            raise Exception("Ising Class: beta must be real. Exception while setting beta!")
    @J.setter
    def J(self,var):
        if type(var) == float or int:
            self.__J = var
        else:
            raise Exception("Ising Class: J must be real. Exception while setting J!")
    
    def update(self):
        DH = self.metropolis_sweep3(self.dat,self.lat.neighbours,self.lat.V,self.beta,self.J)
        return DH

    def H(self):
        H=0.
        for ix in self.lat.sites:
            for neigh in self.lat.get_forward_neighbours(ix):
                H-=self.dat[ix]*self.dat[neigh] 
        
        H *= self.J

        if np.abs(self.B) >1e-14:
            H -= self.B*self.lat

        return H

    def mag(self):
        return np.mean(self.dat)

        
    @staticmethod
    @numba.njit(fastmath=False)
    def metropolis_random_sweep(dat,neighbours_dat,V,beta,J): # this is slower than the systematic sweep.
        DH_acc = 0
        nneigh = 4
        acc_ratio_arr = np.exp(-beta*2*J*np.arange(-nneigh,nneigh+1))
        for _ in range(V):
            # pick up a random site
            ix = np.random.randint(V)

            # compute DH  = H1 - H0 
            DH=0
            
            #for neigh_ix in np.nditer(neighbours_dat[ix,:]): does not speed up
            for neigh_ix in neighbours_dat[ix,:]:
                DH+=dat[neigh_ix] 
            DH*= dat[ix]
            acc_ratio = acc_ratio_arr[DH+nneigh] # speed up by ~10%
            
            if np.random.rand() < min(1,acc_ratio):
                dat[ix] *=-1
                DH*=2*J
                DH_acc+=DH
        
        return DH_acc

    @staticmethod
    @numba.njit(fastmath=False)
    def metropolis_sweep(dat,neighbours_dat,V,beta,J):
        DH_acc = 0
        nneigh = 4
        acc_ratio_arr = np.exp(-beta*2*J*np.arange(-nneigh,nneigh+1))
        for ix in range(V):
            # compute DH  = H1 - H0 
            DH=0
            
            #for neigh_ix in np.nditer(neighbours_dat[ix,:]): does not speed up
            for neigh_ix in neighbours_dat[ix,:]:
                DH+=dat[ix]*dat[neigh_ix] 
            
            acc_ratio = acc_ratio_arr[DH+nneigh] # speed up by ~10%
            
            if np.random.rand() < min(1,acc_ratio):
                dat[ix] *=-1
                DH*=2*J
                DH_acc+=DH
        
        return DH_acc
    
    @staticmethod
    @numba.njit(fastmath=True)
    def metropolis_sweep2(dat,neighbours_dat,V,beta,J):
        DH_acc = 0
        nneigh = len(neighbours_dat[0,:])
        acc_ratio_arr = np.exp(-beta*2*J*np.arange(-nneigh,nneigh+1))
        ix = 0
        for s in np.nditer(dat): # this save 10 %?
            # compute DH  = H1 - H0 
            DH=0
            
            #for neigh_ix in np.nditer(neighbours_dat[ix,:]): does not speed up
            for neigh_ix in neighbours_dat[ix,:]:
                DH+=dat[neigh_ix] 
            DH = s*DH
          
            acc_ratio = acc_ratio_arr[DH+nneigh] # speed up by ~10%
            
            if np.random.rand() < min(1,acc_ratio):
                dat[ix] *=-1
                DH*=2*J
                DH_acc+=DH
            ix+=1
        
        return DH_acc
   
    @staticmethod
    @numba.njit(fastmath=False)
    def metropolis_sweep3(dat,neighbours_dat,V,beta,J):
        DH_acc = 0
        nneigh = len(neighbours_dat[0,:])
        acc_ratio_arr = np.exp(-beta*2*J*np.arange(-nneigh,nneigh+1))
        ix = 0
        for ix,ix_neighbours in gen_neigh(len(dat),neighbours_dat):
            # compute DH  = H1 - H0 
            DH=0
            
            #for neigh_ix in np.nditer(neighbours_dat[ix,:]): does not speed up
            for neigh_ix in ix_neighbours:
                DH+=dat[ix]*dat[neigh_ix] 
          
            acc_ratio = acc_ratio_arr[DH+nneigh] # speed up by ~10%
            
            if np.random.rand() < min(1,acc_ratio):
                dat[ix] *=-1
                DH*=2*J
                DH_acc+=DH
            ix+=1
        
        return DH_acc
    #@staticmethod
    #@numba.njit(fastmath=True)
    def metropolis_sweep4(self,neighbours_dat,V,beta,J):
        DH_acc = 0
        nneigh = len(neighbours_dat[0,:])
        #print(beta)
        #print(J)
        acc_ratio_arr = np.exp(-beta*2*J*np.arange(-nneigh,nneigh+1))
        #print(acc_ratio_arr)
        ix = 0
        for ix_neighbours in neighbours_dat:
            # compute DH  = H1 - H0 
            DH=0
            
            for neigh_ix in ix_neighbours:
                DH+=self.dat[neigh_ix] 
           
            DH = DH*self.dat[ix]
            acc_ratio = acc_ratio_arr[DH+nneigh] # speed up by ~10%
            #print(DH)
            if np.abs(np.exp(-beta*2*J*DH)-acc_ratio_arr[DH+nneigh]) > 1e-7:
                raise Exception("Error acc_ratio")
            self.test_DH(ix)
            
            if np.random.rand() < min(1,acc_ratio):
                #print("conf accepted")

                self.dat[ix] *=-1
                DH*=2*J
                DH_acc+=DH
                #print(self.dat)
                #print(f"conf accepted with DH = {DH}")
               
            ix+=1
        
        return DH_acc


    def test_DH(self,ix):
        H0 = self.H()
        self.dat[ix] *= -1
        H1 = self.H()
        self.dat[ix] *= -1
        DH = 0.
        for neigh_ix in self.lat.neighbours[ix,:]:
                DH+=self.dat[neigh_ix] 
        DH = 2*self.J*DH*self.dat[ix]
        
        if DH != H1-H0:
            raise Exception("Ising: Error in the calculation of DH.")
