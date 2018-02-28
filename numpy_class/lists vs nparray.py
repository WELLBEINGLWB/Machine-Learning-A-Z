import numpy as np

# creating a list
L = [2,3,4]

# append value to list
L.append(L[2])

# lists can be appended
L2 = []
for e in L:
      L2.append(e+e)

# multiplying a list with an int returns appended lists
L = 3*L

# create original list
L = L[0:3]

# numpy array
A = np.array([1,2,3,4,5])

# dot product
dot_A = (A*A).sum()

# elementwise operations
sqrt_A = np.sqrt(A)

# dot product with for loop
a = np.array([1,2])
b = np.array([2,1])
dot = 0
for e, f in zip(a, b):
      dot += e*f
      
dot = sum(a*b)

Amag = np.linalg.norm(A)
