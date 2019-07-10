import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit

"""
Test "Circuit.simulate()"
"""

def test_simulate():
    """
    Test "Circuit.simulate()"
    """
    circuit = simple_circuit(0)
    wfn = circuit.simulate()
    error1 = np.subtract(wfn, [np.sqrt(1/2),0,0,np.sqrt(1/2)])
    error1 = np.sum(np.abs(error1))
    
    circuit = simple_circuit(1)
    wfn = circuit.simulate()
    error2 = np.subtract(wfn, [np.sqrt(1/2),0,0,0,0,0,0,np.sqrt(1/2)])
    error2 = np.sum(np.abs(error2))
    error = error1+error2
    # print(error)
    return error<10-4



if __name__ == "__main__()":
    test_simulate()

# test_simulate()