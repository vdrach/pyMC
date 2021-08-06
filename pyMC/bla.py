import numpy as np

a = np.array([[0,1,2]]).T
b = np.array([[1,2],[2,3],[3,4]])
print(a,b)

#ix, neigh = np.nested_iters([a,b],[[0],[1]])
#print(ix,neigh)
c = zip(a,b)
for ix,neigh in np.nditer(c):
    print("uih")
    print(ix,neigh)
#for ix,neigh in np.ndindex((a,b)):
#    print(ix,neigh)
    #0 and [1,2]
