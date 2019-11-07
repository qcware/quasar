import numpy as np
import vulcan

dim = (2,3,4,5)
A = np.reshape(np.arange(120), dim)
A = np.array(A, dtype=np.float64)
vulcan.array_fool(A)

dim = (2,3,4,5)
A = np.reshape(np.arange(120), dim)
A = 1.j* np.array(A, dtype=np.complex128)
vulcan.array_fool2(A)
print(A)
