import numpy as np

def L1_error(result):
    """
    Input: 
            result (List(Tuple)): A list of tuples. The 1st element of tuple is the predition, the 2nd is the answer.
    Output:
            (Bool): If the L1 error between predictions and answers is smaller than 1e-4, return True.
    """
    error = 0.0
    for a, b in result:
        error += np.linalg.norm(np.subtract(a, b))
        
    return error < 1e-4