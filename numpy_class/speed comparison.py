import numpy as np
from datetime import datetime

a = np.random.random(100)
b = np.random.random(100)
T = 100000

def slow_dot_product(a,b):
      result = 0
      for e, f in zip(a, b):
            result += e*f
      return result

# conducting slow dot product
t0 = datetime.now()
for i in range(T):
      slow_dot_product(a, b)
t1 = datetime.now() - t0

# performing fast dot product
t0 = datetime.now()
for i in range(T):
      a*b
t2 = datetime.now() - t0

print("faster = ", t1/t2)