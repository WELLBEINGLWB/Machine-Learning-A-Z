import numpy as np

# matrix
M = np.array([ [1,2], [3,4] ])

#list of lists
L = [ [1,2], [3,4] ]

l1 = L[0] # first list
l2 = L[1] # second list

l11 = L[0][0] # first entry

m11 = M[0][0]
m11_alt = M[0,0] # less typing for np.array

M2 = np.matrix([ [1,2], [3,4] ]) # not recommended

# creating matrices, arrays

R = np.random.random((2,3))
R.var()
R.mean()


M * R

# more matrix operations
Minv = np.linalg.inv(M)

I = M.dot(Minv)

Mdet = np.linalg.det(M)

Mdiag = np.diag(M)

Mdiag = np.diag(M)

# outer inner product - same same
np.trace(M)
np.diag(M).sum()

b = np.array([2,8])

# solving linear algebra system
x = np.linalg.inv(M).dot(b)
np.linalg.solve(M,b) # always use solve if possible

# example
Mex = np.array([ [1,1], [1.5,4] ])
bex = np.array([2200, 5050])
xex = np.linalg.solve(Mex, bex)
