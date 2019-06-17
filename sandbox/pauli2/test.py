import numpy as np
import quasar

I, X, Y, Z = quasar.Pauli.IXYZ()
O0s = X[0], Y[0], Z[0]
O1s = X[1], Y[1], Z[1]

M0s = quasar.Matrix.X, quasar.Matrix.Y, quasar.Matrix.Z
M1s = M0s

for O0, M0 in zip(O0s, M0s):
    for O1, M1 in zip(O1s, M1s):
        pauli = (O0 * O1)
        matrixA = pauli.compute_hilbert_matrix()
        matrixB = np.kron(M0, M1)
        print(np.max(np.abs(matrixA - matrixB)))
         
        
