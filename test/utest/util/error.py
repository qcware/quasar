import numpy as np

def L1_error(result):
    error = 0.0
    for a, b in result:
        error += np.linalg.norm(np.subtract(a, b))
    return error < 1e-4