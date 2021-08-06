import numpy as np
import numba
from math import prod

# see https://tenpy.github.io/reference/tenpy.models.lattice.Lattice.html
class Lattice:
    def __init__(self,shape,BC,parallel=False,pattern='square'):
        if type(shape) != tuple:
            raise Exception("class Lattice: dim must be a tuple")
        if parallel == True:
            raise Exception("class Lattice: parallel version not implemented !")
        if pattern != 'square' :
            raise Exception("class Lattice: Only pattern='square' is implemented !")

        self.shape=shape
        self.V = prod(shape)
        self.dim = len(shape)
        
        self.init_basis_vec()
        self.init_neighbours()
        
        print(f"class Lattice: initialising a {pattern} lattice  of shape {shape} with BC: {BC}")
        # the type of the indices should be inferred from V: no need to use np.int64 on small lattices... should throw error if grid is too large.
        ix_ = np.zeros(self.V,dtype=np.int64)
        
        self.sites = range(self.V)

    def init_basis_vec(self):
        self.basis_vec = np.zeros((self.dim,self.dim),np.int64)
        for d in range(self.dim):
            self.basis_vec[d,d] = 1
    
            
    def ispos_valid(pos):
        if len(pos) != self.dim:
            raise Exception("class Lattice: position of wrong dimension")

        for i in range(self.dim):
            if pos[i] >= self.shape[i]:
                raise Exception("class Lattice: position invalid: pos[{i}]= {pos[i]} and shape[{i}] = {self.shape[i]}")

    #index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3;
    def get_ix(self,pos):
        ix=0
        w = 1
        for i in range(self.dim):
            ix += pos[i]*w
            w *= self.shape[i]

        return ix

    #x = Index % D0;
    #y = ( ( Index - x ) / D0 ) %  D1;
    #z = ( ( Index - y * D0 - x ) / (D0 * D1) ) % D2; 
    #t = ( ( Index - z * D1 * D0 - y * D0 - x ) / (D0 * D1 * D2) ) % D3; 
    def get_pos(self,ix):
        pos = np.zeros(self.dim,dtype=np.int64)
        shift = 0
        div = 1
        w = 1
        for i in range(self.dim):
            pos[i] = ((ix - shift)/div) % self.shape[i] 
            #print(f"ix : {ix} shift {shift} div {div} shape {self.shape[i]} {pos[i]}")
            
            div *= self.shape[i]
            shift += pos[i]*w
            w *= self.shape[i]
        return pos

    def get_forward_neighbours(self,ix):
        res = np.zeros(self.dim,dtype=np.int64)
        pos=self.get_pos(ix)
        for d in range(0,self.dim):
            pos_shifted = (pos + self.basis_vec[d,:])%self.shape
            res[d] = self.get_ix(pos_shifted)
        
        return res
    def init_neighbours(self):
        self.neighbours = np.zeros((self.V,2*self.dim),dtype=np.int64)
        for ix in range(self.V):
            for i in range(len(self.get_neighbours(ix))):
                self.neighbours[ix,i] = self.get_neighbours(ix)[i]

    def get_neighbours(self,ix):
        res = np.zeros(2*self.dim,dtype=np.int64)
        pos=self.get_pos(ix)
        i = 0
        for d in range(0,self.dim):
            pos_shifted = (pos + self.basis_vec[d,:])%self.shape
            res[i] = self.get_ix(pos_shifted)
            pos_shifted = (pos - self.basis_vec[d,:])%self.shape
            res[i+1] = self.get_ix(pos_shifted)
            i+=2
            
        return res

    def print_dict_ix_pos(self):
        for ix in range(self.V): # loop over the bulk
            pos=self.get_pos(ix)
            check_ix = self.get_ix(pos)
            print(f"ix = {ix} -> pos = {pos} -> ix = {check_ix}")
            if check_ix != ix :
                raise Exception("class Lattice: indexing inconsistent!")
        

@numba.njit(fastmath=True)
def gen_neigh(V,neighbours):
        ix = 0 
        while ix < V:
            yield ix,neighbours[ix,:]
            ix+=1
